#include "GltfScene.h"
#include "GltfConvert.h"
#include "project/Assets.h"
#include "state/Scene.h"

#include "File.h"
#include "Path.h"
#include "Profile.h"
#include "TransformMath.h"
#include "Variant.h"
#include "animation/AnimationData.h"
#include "animation/AnimationTimeline.h"
#include "animation/MorphWeightState.h"
#include "armature/Armature.h"
#include "armature/ArmatureComponents.h"
#include "audio/AcousticMaterial.h"
#include "audio/AudioSystem.h"
#include "audio/AudioTypes.h"
#include "audio/ContactModel.h"
#include "audio/ContactSurface.h"
#include "audio/ModalModes.h"
#include "mesh/MeshAttributes.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshCreate.h"
#include "mesh/MeshStore.h"
#include "object/ObjectOps.h"
#include "physics/PhysicsTypes.h"
#include "render/GpuBuffers.h"
#include "render/Instance.h"
#include "render/MaterialComponents.h"
#include "render/PbrFeature.h"
#include "render/Textures.h"
#include "scene/Entity.h"
#include "scene/SceneGraph.h"
#include "scene/SceneGraphOps.h"
#include "scene/WorldTransform.h"
#include "viewport/ViewportDisplay.h"

#include "meshoptimizer.h"

#include "numeric/FastGltf.h"
#include <fastgltf/core.hpp>
#include <simdjson.h>

#include <bit>
#include <numbers>
#include <numeric>
#include <unordered_set>

namespace gltf {
using namespace detail;
namespace {
void CollectExtras(simdjson::dom::object *extras, size_t idx, fastgltf::Category cat, void *userPtr) {
    if (!extras || !userPtr) return;
    static_cast<ExtrasMap *>(userPtr)->emplace(ExtrasKey(uint32_t(cat), idx), simdjson::minify(*extras));
}
std::optional<uint32_t> ToIndex(size_t index, size_t upper_bound) {
    if (index >= upper_bound) return {};
    return index;
}
std::optional<uint32_t> ToIndex(const fastgltf::Optional<size_t> &index, size_t upper_bound) {
    if (!index) return {};
    return ToIndex(*index, upper_bound);
}
std::optional<Filter> ToFilter(const fastgltf::Optional<fastgltf::Filter> &filter) {
    if (!filter) return {};
    return detail::ToFilter(*filter);
}

vec2 ToVec2(const fastgltf::math::nvec2 &v) { return std::bit_cast<vec2>(v); }
vec3 ToVec3(const fastgltf::math::nvec3 &v) { return std::bit_cast<vec3>(v); }
vec4 ToVec4(const fastgltf::math::nvec4 &v) { return std::bit_cast<vec4>(v); }
quat ToQuat(const fastgltf::math::fquat &q) { return std::bit_cast<quat>(q); }
Transform TrsToTransform(const fastgltf::TRS &trs) { return {.P = ToVec3(trs.translation), .R = numeric::Normalize(ToQuat(trs.rotation)), .S = ToVec3(trs.scale)}; }

// Slot initially contains a glTF texture index and later contains a bindless Scene.cpp slot.
// Supply meta for top-level material textures that require texCoord override round trips.
template<typename OptT>
::TextureInfo ToTextureIndex(const OptT &opt, const fastgltf::Asset &asset, TextureTransformMeta *meta = nullptr) {
    if (!opt) return {};
    const auto texture_index = ToIndex(opt->textureIndex, asset.textures.size());
    if (!texture_index) return {};
    ::TextureInfo out{.Slot = *texture_index, .TexCoord = uint32_t(opt->texCoordIndex)};
    if (meta) meta->SourceBaseTexCoord = uint32_t(opt->texCoordIndex);
    if (opt->transform) {
        if (meta) meta->SourceHadExtension = true;
        out.UvRotation = opt->transform->rotation;
        out.UvOffset = ToVec2(opt->transform->uvOffset);
        out.UvScale = ToVec2(opt->transform->uvScale);
        if (const auto tc_override = ToIndex(opt->transform->texCoordIndex, std::numeric_limits<uint32_t>::max())) {
            if (meta) meta->SourceTexCoordOverride = tc_override;
            out.TexCoord = *tc_override;
        }
    }
    return out;
}

std::expected<Image, std::string> ReadImage(const fastgltf::Asset &asset, uint32_t image_index, const std::filesystem::path &base_dir) {
    if (image_index >= asset.images.size()) return std::unexpected{std::format("glTF image index {} is out of range.", image_index)};
    const auto &image = asset.images[image_index];

    Image out{.Bytes = {}, .MimeType = MimeType::None, .Source = Image::SourceKind::Embedded, .SourceHadMimeType = false, .IsDirty = false, .Name = std::string{image.name}, .Uri = {}, .SourcePath = {}};
    const auto copy_bytes = [&out](const auto &data, fastgltf::MimeType mime_type) {
        out.Bytes.resize(data.size());
        std::memcpy(out.Bytes.data(), data.data(), data.size());
        out.MimeType = ToMimeType(mime_type);
        out.SourceHadMimeType = mime_type != fastgltf::MimeType::None;
    };
    if (const auto *uri = std::get_if<fastgltf::sources::URI>(&image.data)) {
        if (!uri->uri.isLocalPath()) return std::unexpected{std::format("glTF image {} URI '{}' is not a local path.", image_index, uri->uri.string())};
        auto image_path = uri->uri.fspath();
        if (image_path.is_relative()) image_path = base_dir / image_path;
        image_path = image_path.lexically_normal();
        auto bytes = File::Read(image_path);
        if (!bytes) return std::unexpected{std::move(bytes.error())};
        out.Bytes = std::move(*bytes);
        out.MimeType = ToMimeType(uri->mimeType);
        out.Source = Image::SourceKind::External;
        out.SourceHadMimeType = uri->mimeType != fastgltf::MimeType::None;
        out.Uri = uri->uri.string();
        out.SourcePath = image_path.string();
    } else if (const auto *buffer_view = std::get_if<fastgltf::sources::BufferView>(&image.data)) {
        if (buffer_view->bufferViewIndex >= asset.bufferViews.size()) {
            return std::unexpected{std::format("glTF image {} references invalid bufferView index {}.", image_index, buffer_view->bufferViewIndex)};
        }
        copy_bytes(fastgltf::DefaultBufferDataAdapter{}(asset, buffer_view->bufferViewIndex), buffer_view->mimeType);
    } else if (const auto *array = std::get_if<fastgltf::sources::Array>(&image.data)) {
        // The parser decodes data URIs into an array, the only in-memory source with external image loading off.
        copy_bytes(array->bytes, array->mimeType);
        out.Source = Image::SourceKind::DataUri;
    } else {
        return std::unexpected{std::format("glTF image {} has no supported data source.", image_index)};
    }
    if (out.MimeType == MimeType::None) out.MimeType = SniffMimeType(out.Bytes);
    return out;
}

// Append `count` vertices of the optional `name` attribute, backfilling with `fill` where absent.
// Sets `bit` in `flags` when present. With `check_count`, a count mismatch with POSITION is an error.
template<typename T>
std::expected<void, std::string> AppendVertexAttr(
    const fastgltf::Asset &asset, const fastgltf::Primitive &primitive, std::string_view name,
    std::optional<std::vector<T>> &attr, uint32_t base_vertex, size_t count, T fill,
    uint32_t *flags = nullptr, uint32_t bit = 0, bool check_count = false
) {
    const auto *const it = primitive.findAttribute(name);
    if (it == primitive.attributes.end()) {
        if (attr) attr->resize(base_vertex + count, fill);
        return {};
    }
    if (flags) *flags |= bit;
    if (!attr) {
        attr.emplace();
        attr->resize(base_vertex, fill);
    }
    const auto &accessor = asset.accessors[it->accessorIndex];
    if (check_count && accessor.count != count) {
        return std::unexpected{std::format("glTF primitive {} count ({}) must match POSITION count ({}).", name, accessor.count, count)};
    }
    attr->resize(base_vertex + count, fill);
    fastgltf::copyFromAccessor<T>(asset, accessor, &(*attr)[base_vertex]);
    return {};
}

// CPU stores vec4 regardless of source; track whether any primitive used VEC3.
// With `flags`, a present accessor with a mismatched count or a non-VEC3/VEC4 type is an error.
std::expected<void, std::string> AppendColor0(
    const fastgltf::Asset &asset, const fastgltf::Primitive &primitive, ::MeshVertexAttributes &attrs,
    uint32_t base_vertex, size_t count, uint32_t *flags = nullptr
) {
    const auto *const color_it = primitive.findAttribute("COLOR_0");
    if (color_it == primitive.attributes.end()) {
        if (attrs.Colors0) attrs.Colors0->resize(base_vertex + count, vec4{1.f});
        return {};
    }
    if (flags) *flags |= MeshAttributeBit_Color0;
    if (!attrs.Colors0) {
        attrs.Colors0.emplace();
        attrs.Colors0->resize(base_vertex, vec4{1.f});
    }
    const auto &color_accessor = asset.accessors[color_it->accessorIndex];
    if (flags && color_accessor.count != count) {
        return std::unexpected{std::format("glTF primitive COLOR_0 count ({}) must match POSITION count ({}).", color_accessor.count, count)};
    }
    attrs.Colors0->resize(base_vertex + count, vec4{1.f});
    if (color_accessor.type == fastgltf::AccessorType::Vec3) {
        if (attrs.Colors0ComponentCount == 0) attrs.Colors0ComponentCount = 3;
        std::vector<vec3> colors(count);
        fastgltf::copyFromAccessor<vec3>(asset, color_accessor, colors.data());
        for (uint32_t i = 0; i < count; ++i) (*attrs.Colors0)[base_vertex + i] = vec4{colors[i], 1.f};
    } else if (color_accessor.type == fastgltf::AccessorType::Vec4) {
        attrs.Colors0ComponentCount = 4;
        fastgltf::copyFromAccessor<vec4>(asset, color_accessor, &(*attrs.Colors0)[base_vertex]);
    } else if (flags) {
        return std::unexpected{std::format("glTF primitive COLOR_0 accessor type must be VEC3 or VEC4, got type {}.", int(color_accessor.type))};
    }
    return {};
}

// Primitive indices, or iota over the vertex range when the primitive is non-indexed.
std::vector<uint32_t> ReadIndices(const fastgltf::Asset &asset, const fastgltf::Primitive &primitive, size_t vertex_count) {
    std::vector<uint32_t> indices;
    if (primitive.indicesAccessor) {
        const auto &index_accessor = asset.accessors[*primitive.indicesAccessor];
        indices.resize(index_accessor.count);
        fastgltf::copyFromAccessor<uint32_t>(asset, index_accessor, indices.data());
    } else {
        indices.resize(vertex_count);
        std::iota(indices.begin(), indices.end(), 0u);
    }
    return indices;
}

// Appends a line or point primitive while preserving channel alignment across merged primitives.
void AppendNonTrianglePrimitive(const fastgltf::Asset &asset, const fastgltf::Primitive &primitive, ::MeshData &target, ::MeshVertexAttributes &attrs) {
    const auto *const position_it = primitive.findAttribute("POSITION");
    if (position_it == primitive.attributes.end()) return;

    const auto &position_accessor = asset.accessors[position_it->accessorIndex];
    if (position_accessor.count == 0) return;

    const uint32_t base_vertex = target.Positions.size();
    target.Positions.resize(base_vertex + position_accessor.count);
    fastgltf::copyFromAccessor<vec3>(asset, position_accessor, &target.Positions[base_vertex]);

    std::ignore = AppendVertexAttr(asset, primitive, "NORMAL", attrs.Normals, base_vertex, position_accessor.count, vec3{0.f});
    std::ignore = AppendColor0(asset, primitive, attrs, base_vertex, position_accessor.count);

    if (primitive.type == fastgltf::PrimitiveType::Points) return;

    const auto indices = ReadIndices(asset, primitive, position_accessor.count);

    switch (primitive.type) {
        case fastgltf::PrimitiveType::Points:
        case fastgltf::PrimitiveType::Triangles:
        case fastgltf::PrimitiveType::TriangleStrip:
        case fastgltf::PrimitiveType::TriangleFan:
            break;
        case fastgltf::PrimitiveType::Lines:
            for (uint32_t i = 0; i + 1 < indices.size(); i += 2) target.Edges.emplace_back(std::array{base_vertex + indices[i], base_vertex + indices[i + 1]});
            break;
        case fastgltf::PrimitiveType::LineStrip:
            for (uint32_t i = 0; i + 1 < indices.size(); ++i) target.Edges.emplace_back(std::array{base_vertex + indices[i], base_vertex + indices[i + 1]});
            break;
        case fastgltf::PrimitiveType::LineLoop:
            for (uint32_t i = 0; i + 1 < indices.size(); ++i) target.Edges.emplace_back(std::array{base_vertex + indices[i], base_vertex + indices[i + 1]});
            if (indices.size() >= 2) target.Edges.emplace_back(std::array{base_vertex + indices.back(), base_vertex + indices.front()});
            break;
    }
}

bool IsTriangleType(fastgltf::PrimitiveType type) {
    return type == fastgltf::PrimitiveType::Triangles || type == fastgltf::PrimitiveType::TriangleStrip || type == fastgltf::PrimitiveType::TriangleFan;
}

std::expected<void, std::string> AppendPrimitive(
    const fastgltf::Asset &asset,
    const fastgltf::Primitive &primitive,
    ::MeshData &mesh,
    ::MeshVertexAttributes &attrs,
    std::optional<ArmatureDeformData> &deform,
    std::optional<MorphTargetData> &morph,
    uint32_t total_vertex_count,
    uint32_t &attribute_flags
) {
    if (!IsTriangleType(primitive.type)) return {};

    const auto *const position_it = primitive.findAttribute("POSITION");
    if (position_it == primitive.attributes.end()) return {};

    const bool has_joints = primitive.findAttribute("JOINTS_0") != primitive.attributes.end();
    const bool has_weights = primitive.findAttribute("WEIGHTS_0") != primitive.attributes.end();
    if (has_joints != has_weights) return std::unexpected{"glTF primitive has JOINTS_0 without WEIGHTS_0 (or vice versa)."};

    const auto &position_accessor = asset.accessors[position_it->accessorIndex];
    if (position_accessor.count == 0) return {};

    const uint32_t base_vertex = mesh.Positions.size();
    mesh.Positions.resize(base_vertex + position_accessor.count);
    fastgltf::copyFromAccessor<vec3>(asset, position_accessor, &mesh.Positions[base_vertex]);

    const auto vertex_count = position_accessor.count;
    if (auto result = AppendVertexAttr(asset, primitive, "NORMAL", attrs.Normals, base_vertex, vertex_count, vec3{0.f}, &attribute_flags, MeshAttributeBit_Normal); !result) return result;
    if (auto result = AppendVertexAttr(asset, primitive, "TANGENT", attrs.Tangents, base_vertex, vertex_count, vec4{0.f, 0.f, 0.f, 1.f}, &attribute_flags, MeshAttributeBit_Tangent, true); !result) return result;
    if (auto result = AppendColor0(asset, primitive, attrs, base_vertex, vertex_count, &attribute_flags); !result) return result;

    const std::array uv_sets{&attrs.TexCoords0, &attrs.TexCoords1, &attrs.TexCoords2, &attrs.TexCoords3};
    for (uint32_t set_index = 0; set_index < uv_sets.size(); ++set_index) {
        if (auto result = AppendVertexAttr(asset, primitive, std::format("TEXCOORD_{}", set_index), *uv_sets[set_index], base_vertex, vertex_count, vec2{0.f}, &attribute_flags, MeshAttributeBit_TexCoord0 << set_index, true); !result) return result;
    }

    if (has_joints && !deform) deform.emplace();
    const bool mesh_has_skin = deform && (!deform->Joints.empty() || !deform->Weights.empty());
    if (mesh_has_skin &&
        (deform->Joints.size() != base_vertex ||
         deform->Weights.size() != base_vertex)) {
        return std::unexpected{"glTF primitive append encountered inconsistent skin channel sizes."};
    }

    if (has_joints || mesh_has_skin) {
        deform->Joints.resize(mesh.Positions.size(), uvec4{0});
        deform->Weights.resize(mesh.Positions.size(), vec4{0});
    }

    if (has_joints) {
        // Collect and validate all skin influence accessor pairs (JOINTS_n/WEIGHTS_n).
        std::vector<std::pair<const fastgltf::Accessor *, const fastgltf::Accessor *>> influence_accessors;
        for (uint32_t set_index = 0;; ++set_index) {
            const auto j_name = std::format("JOINTS_{}", set_index);
            const auto w_name = std::format("WEIGHTS_{}", set_index);
            const auto *const j_it = primitive.findAttribute(j_name);
            const auto *const w_it = primitive.findAttribute(w_name);
            if (j_it == primitive.attributes.end() && w_it == primitive.attributes.end()) break;
            if ((j_it == primitive.attributes.end()) != (w_it == primitive.attributes.end())) {
                return std::unexpected{std::format("glTF primitive has {} without {} (or vice versa).", j_name, w_name)};
            }

            const auto &j_acc = asset.accessors[j_it->accessorIndex];
            const auto &w_acc = asset.accessors[w_it->accessorIndex];
            if (j_acc.count != position_accessor.count || w_acc.count != position_accessor.count) {
                return std::unexpected{std::format(
                    "glTF primitive skin attribute counts must match POSITION count (POSITION={}, {}={}, {}={}).",
                    position_accessor.count, j_name, j_acc.count, w_name, w_acc.count
                )};
            }
            influence_accessors.emplace_back(&j_acc, &w_acc);
        }

        if (influence_accessors.size() == 1) {
            fastgltf::copyFromAccessor<uvec4>(asset, *influence_accessors.front().first, &deform->Joints[base_vertex]);
            fastgltf::copyFromAccessor<vec4>(asset, *influence_accessors.front().second, &deform->Weights[base_vertex]);
        } else {
            struct InfluenceSet {
                std::vector<uvec4> Joints;
                std::vector<vec4> Weights;
            };
            std::vector<InfluenceSet> sets(influence_accessors.size());
            for (size_t set_index = 0; set_index < sets.size(); ++set_index) {
                auto &s = sets[set_index];
                const auto [j_acc, w_acc] = influence_accessors[set_index];
                s.Joints.resize(position_accessor.count);
                fastgltf::copyFromAccessor<uvec4>(asset, *j_acc, s.Joints.data());
                s.Weights.resize(position_accessor.count);
                fastgltf::copyFromAccessor<vec4>(asset, *w_acc, s.Weights.data());
            }

            // Multiple influence sets: merge all, keep top 4 by weight, renormalize.
            // glTF 2.0 section 3.7.3.1 permits implementations to support four influences.
            // Reference: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#skinned-mesh-attributes.
            const auto total_influences = sets.size() * 4u;
            std::vector<std::pair<uint32_t, float>> all(total_influences);
            const auto top4_end = all.begin() + 4;
            const auto by_weight = [](const auto &a, const auto &b) { return a.second > b.second; };
            for (size_t i = 0; i < position_accessor.count; ++i) {
                uint32_t n = 0;
                for (const auto &s : sets) {
                    const auto &j = s.Joints[i];
                    const auto &w = s.Weights[i];
                    all[n++] = {j.x, w.x};
                    all[n++] = {j.y, w.y};
                    all[n++] = {j.z, w.z};
                    all[n++] = {j.w, w.w};
                }
                std::partial_sort(all.begin(), top4_end, all.begin() + n, by_weight);

                const float sum = all[0].second + all[1].second + all[2].second + all[3].second;
                const float inv = sum > 0.f ? 1.f / sum : 0.f;
                deform->Joints[base_vertex + i] = {all[0].first, all[1].first, all[2].first, all[3].first};
                deform->Weights[base_vertex + i] = {all[0].second * inv, all[1].second * inv, all[2].second * inv, all[3].second * inv};
            }
        }
    }

    // Morph deltas are target-major over the whole mesh, so each primitive writes its vertex range of every target in place.
    if (!primitive.targets.empty()) {
        const uint32_t target_count = primitive.targets.size();
        if (!morph) {
            morph.emplace();
            morph->TargetCount = target_count;
            morph->PositionDeltas.assign(size_t(target_count) * total_vertex_count, vec3{0.f});
        }
        if (morph->TargetCount != target_count) return std::unexpected{"glTF primitive morph target count mismatch between primitives of the same mesh."};

        const auto any_target_has = [&](std::string_view name) {
            for (uint32_t t = 0; t < target_count; ++t) {
                if (primitive.findTargetAttribute(t, name) != primitive.targets[t].end()) return true;
            }
            return false;
        };
        if (morph->NormalDeltas.empty() && any_target_has("NORMAL")) morph->NormalDeltas.assign(size_t(target_count) * total_vertex_count, vec3{0.f});
        if (morph->TangentDeltas.empty() && any_target_has("TANGENT")) morph->TangentDeltas.assign(size_t(target_count) * total_vertex_count, vec3{0.f});
        const auto copy_target = [&](uint32_t t, std::string_view name, std::vector<vec3> &channel) {
            if (channel.empty()) return;
            const auto *it = primitive.findTargetAttribute(t, name);
            if (it == primitive.targets[t].end()) return;
            const auto &accessor = asset.accessors[it->accessorIndex];
            if (accessor.count == vertex_count) fastgltf::copyFromAccessor<vec3>(asset, accessor, &channel[size_t(t) * total_vertex_count + base_vertex]);
        };
        for (uint32_t t = 0; t < target_count; ++t) {
            copy_target(t, "POSITION", morph->PositionDeltas);
            copy_target(t, "NORMAL", morph->NormalDeltas);
            copy_target(t, "TANGENT", morph->TangentDeltas);
        }
    }

    // Process complete triangle lists as one shifted index block.
    if (primitive.type == fastgltf::PrimitiveType::Triangles) {
        const auto index_count = primitive.indicesAccessor ? asset.accessors[*primitive.indicesAccessor].count : position_accessor.count;
        if (index_count >= 3 && index_count % 3 == 0) {
            const auto corners = mesh.AddTriangleCorners(uint32_t(index_count));
            if (primitive.indicesAccessor) {
                fastgltf::copyFromAccessor<uint32_t>(asset, asset.accessors[*primitive.indicesAccessor], corners.data());
                if (base_vertex != 0) {
                    for (auto &corner : corners) corner += base_vertex;
                }
            } else {
                std::iota(corners.begin(), corners.end(), base_vertex);
            }
            return {};
        }
    }

    const auto indices = ReadIndices(asset, primitive, position_accessor.count);
    if (indices.size() < 3) return {};

    const auto add_triangle = [&mesh](uint32_t a, uint32_t b, uint32_t c) {
        mesh.AddFace(std::array{a, b, c});
    };
    if (primitive.type == fastgltf::PrimitiveType::TriangleStrip) {
        mesh.ReserveFaces(uint32_t(indices.size()) - 2u, 3u);
        for (uint32_t i = 0; i + 2 < indices.size(); ++i) {
            if (i % 2 == 0) add_triangle(base_vertex + indices[i], base_vertex + indices[i + 1], base_vertex + indices[i + 2]);
            else add_triangle(base_vertex + indices[i + 1], base_vertex + indices[i], base_vertex + indices[i + 2]);
        }
    } else if (primitive.type == fastgltf::PrimitiveType::TriangleFan) {
        mesh.ReserveFaces(uint32_t(indices.size()) - 2u, 3u);
        for (uint32_t i = 1; i + 1 < indices.size(); ++i) {
            add_triangle(base_vertex + indices[0], base_vertex + indices[i], base_vertex + indices[i + 1]);
        }
    } else {
        mesh.ReserveFaces(uint32_t(indices.size()) / 3u, 3u);
        for (uint32_t i = 0; i + 2 < indices.size(); i += 3) {
            add_triangle(base_vertex + indices[i], base_vertex + indices[i + 1], base_vertex + indices[i + 2]);
        }
    }
    return {};
}

// Raw bytes of a loaded buffer. Meshopt compressed data is referenced by buffer index, not bufferView.
std::optional<std::span<const std::byte>> BufferBytes(const fastgltf::Buffer &buffer) {
    return std::visit(
        fastgltf::visitor{
            [](const fastgltf::sources::Array &a) -> std::optional<std::span<const std::byte>> { return std::span{a.bytes.data(), a.bytes.size()}; },
            [](const fastgltf::sources::Vector &v) -> std::optional<std::span<const std::byte>> { return std::span{v.bytes.data(), v.bytes.size()}; },
            [](const fastgltf::sources::ByteView &b) -> std::optional<std::span<const std::byte>> { return b.bytes; },
            [](const auto &) -> std::optional<std::span<const std::byte>> { return std::nullopt; },
        },
        buffer.data
    );
}

// Decodes each meshopt-compressed bufferView into a new buffer for ordinary accessor reads.
std::expected<void, std::string> DecodeMeshoptCompression(fastgltf::Asset &asset) {
    using fastgltf::MeshoptCompressionMode, fastgltf::MeshoptCompressionFilter;
    for (auto &view : asset.bufferViews) {
        if (!view.meshoptCompression) continue;
        const auto &comp = *view.meshoptCompression;
        if (comp.bufferIndex >= asset.buffers.size()) return std::unexpected{"references an invalid buffer index"};
        const auto source = BufferBytes(asset.buffers[comp.bufferIndex]);
        if (!source) return std::unexpected{"source buffer has no readable data"};
        if (comp.byteOffset + comp.byteLength > source->size()) return std::unexpected{"source range is out of bounds"};

        const auto *src = reinterpret_cast<const unsigned char *>(source->data() + comp.byteOffset);
        std::vector<std::byte> decoded(comp.count * comp.byteStride);
        auto *dst = reinterpret_cast<unsigned char *>(decoded.data());
        const int rc = [&] {
            switch (comp.mode) {
                case MeshoptCompressionMode::Attributes: return meshopt_decodeVertexBuffer(dst, comp.count, comp.byteStride, src, comp.byteLength);
                case MeshoptCompressionMode::Triangles: return meshopt_decodeIndexBuffer(dst, comp.count, comp.byteStride, src, comp.byteLength);
                case MeshoptCompressionMode::Indices: return meshopt_decodeIndexSequence(dst, comp.count, comp.byteStride, src, comp.byteLength);
            }
            return -1;
        }();
        if (rc != 0) return std::unexpected{"decode failed"};

        switch (comp.filter) {
            case MeshoptCompressionFilter::None: break;
            case MeshoptCompressionFilter::Octahedral: meshopt_decodeFilterOct(dst, comp.count, comp.byteStride); break;
            case MeshoptCompressionFilter::Quaternion: meshopt_decodeFilterQuat(dst, comp.count, comp.byteStride); break;
            case MeshoptCompressionFilter::Exponential: meshopt_decodeFilterExp(dst, comp.count, comp.byteStride); break;
            case MeshoptCompressionFilter::Color: meshopt_decodeFilterColor(dst, comp.count, comp.byteStride); break;
        }

        const auto decoded_length = decoded.size();
        auto &new_buffer = asset.buffers.emplace_back();
        new_buffer.byteLength = decoded_length;
        new_buffer.data = fastgltf::sources::Array{fastgltf::StaticVector<std::byte>::fromVector(std::move(decoded))};
        view.bufferIndex = asset.buffers.size() - 1;
        view.byteOffset = 0;
        view.byteLength = decoded_length;
        view.meshoptCompression = nullptr;
    }
    return {};
}

// A bare filename has an empty parent, which the parser rejects as the resource directory, so a relative scene path resolves against the working directory.
std::filesystem::path AbsoluteScenePath(const std::filesystem::path &p) {
    return p.is_absolute() ? p : std::filesystem::absolute(p);
}

std::expected<fastgltf::Asset, std::string> ParseAsset(const std::filesystem::path &given_path, ExtrasMap *extras_out = nullptr) {
    const auto path = AbsoluteScenePath(given_path);
    if (std::error_code ec; !std::filesystem::exists(path, ec)) {
        return std::unexpected{std::string{"File does not exist"}};
    }
    auto gltf_file = fastgltf::MappedGltfFile::FromPath(path);
    if (gltf_file.error() != fastgltf::Error::None) return std::unexpected{std::format("Failed to open glTF: {}", fastgltf::getErrorMessage(gltf_file.error()))};

    static constexpr auto EnabledExtensions = fastgltf::Extensions::KHR_mesh_quantization | fastgltf::Extensions::EXT_meshopt_compression | fastgltf::Extensions::KHR_meshopt_compression | fastgltf::Extensions::EXT_mesh_gpu_instancing | fastgltf::Extensions::KHR_lights_punctual | fastgltf::Extensions::EXT_lights_image_based | fastgltf::Extensions::KHR_texture_transform | fastgltf::Extensions::KHR_materials_emissive_strength | fastgltf::Extensions::KHR_materials_unlit | fastgltf::Extensions::KHR_texture_basisu | fastgltf::Extensions::EXT_texture_webp | fastgltf::Extensions::KHR_materials_specular | fastgltf::Extensions::KHR_materials_sheen | fastgltf::Extensions::KHR_materials_ior | fastgltf::Extensions::KHR_materials_dispersion | fastgltf::Extensions::KHR_materials_transmission | fastgltf::Extensions::KHR_materials_diffuse_transmission | fastgltf::Extensions::KHR_materials_volume | fastgltf::Extensions::KHR_materials_clearcoat | fastgltf::Extensions::KHR_materials_anisotropy | fastgltf::Extensions::KHR_materials_iridescence | fastgltf::Extensions::KHR_materials_variants | fastgltf::Extensions::KHR_node_visibility | fastgltf::Extensions::KHR_implicit_shapes | fastgltf::Extensions::KHR_physics_rigid_bodies | fastgltf::Extensions::KHR_audio_rigid_bodies;
    fastgltf::Parser parser{EnabledExtensions};
    if (extras_out) {
        parser.setUserPointer(extras_out);
        parser.setExtrasParseCallback(CollectExtras);
    }
    using fastgltf::Options;
    // LoadExternalImages off so external so ReadImage can preserve the URI for round-trip.
    // GenerateMeshIndices off to synthesize iota locally and track per-primitive presence so non-indexed primitives round-trip.
    static constexpr auto ParseOptions = Options::AllowDouble | Options::LoadExternalBuffers;
    auto parsed = parser.loadGltf(gltf_file.get(), path.parent_path(), ParseOptions);
    if (parsed.error() != fastgltf::Error::None) {
        if (parsed.error() == fastgltf::Error::MissingExtensions) {
            // Reparse with all extensions to recover extensionsRequired after fastgltf aborts the first pass.
            gltf_file.get().reset();
            fastgltf::Parser probe{fastgltf::Extensions(~0U)};
            if (auto probed = probe.loadGltf(gltf_file.get(), path.parent_path(), Options::DontRequireValidAssetMember | Options::AllowDouble);
                probed.error() == fastgltf::Error::None) {
                const auto enabled = fastgltf::stringifyExtensionBits(EnabledExtensions);
                std::string missing;
                for (const auto &req : probed.get().extensionsRequired) {
                    if (!std::ranges::any_of(enabled, [&](const auto &n) { return n == req; })) {
                        if (!missing.empty()) missing += ", ";
                        missing += req;
                    }
                }
                if (!missing.empty()) return std::unexpected{std::format("Missing required extensions: {}", missing)};
            }
        }
        return std::unexpected{std::format("Failed to parse glTF: {}", fastgltf::getErrorMessage(parsed.error()))};
    }

    auto &asset = parsed.get();
    if (auto decoded = DecodeMeshoptCompression(asset); !decoded) {
        return std::unexpected{std::format("Failed to decode meshopt compression: {}", decoded.error())};
    }
    return std::move(asset);
}

// One source mesh's triangle, line, and point parts, with the per-primitive layout every part shares.
struct SourceMesh {
    std::optional<MeshSource> Triangles, Lines, Points;
    MeshSourceLayout Layout;
    bool HasParts() const { return Triangles || Lines || Points; }
};

// Reads every primitive of asset.meshes[source_mesh_index]. Materials index the source materials plus the trailing implicit default.
std::expected<SourceMesh, std::string> ReadSourceMesh(const fastgltf::Asset &asset, uint32_t source_mesh_index) {
    const auto &source_mesh = asset.meshes[source_mesh_index];
    const auto primitive_count = uint32_t(source_mesh.primitives.size());
    const auto material_count = uint32_t(asset.materials.size()) + 1u;
    MeshSourceLayout layout{
        .AttributeFlags = std::vector<uint32_t>(primitive_count, 0u),
        .HasSourceIndices = std::vector<uint8_t>(primitive_count, 0u),
        .DefaultMaterials = std::vector<uint32_t>(primitive_count, material_count - 1u),
        .VariantMappings = std::vector<std::vector<std::optional<uint32_t>>>(primitive_count),
        .Colors0ComponentCount = 0,
        .MorphTangentDeltas = {},
        .Index = source_mesh_index,
        .Kind = MeshKind::Triangles,
        .Name = std::string{source_mesh.name},
    };
    // Morph deltas are written target-major, so the triangle vertex total is fixed before the first primitive appends.
    uint32_t triangle_vertex_count = 0;
    for (const auto &primitive : source_mesh.primitives) {
        if (!IsTriangleType(primitive.type)) continue;
        if (const auto *it = primitive.findAttribute("POSITION"); it != primitive.attributes.end()) triangle_vertex_count += asset.accessors[it->accessorIndex].count;
    }

    ::MeshData triangles, lines, points;
    ::MeshVertexAttributes triangle_attrs, line_attrs, point_attrs;
    std::optional<ArmatureDeformData> deform;
    std::optional<MorphTargetData> morph;
    // Every drawn element records its source primitive, per face for triangles and per vertex for the merged line and point meshes.
    std::vector<uint32_t> face_primitives, line_primitives, point_primitives;
    for (uint32_t primitive_index = 0; primitive_index < primitive_count; ++primitive_index) {
        const auto &primitive = source_mesh.primitives[primitive_index];
        if (const auto material_index = ToIndex(primitive.materialIndex, material_count)) layout.DefaultMaterials[primitive_index] = *material_index;
        layout.HasSourceIndices[primitive_index] = primitive.indicesAccessor.has_value() ? 1u : 0u;
        auto &mappings = layout.VariantMappings[primitive_index];
        mappings.reserve(primitive.mappings.size());
        for (const auto &m : primitive.mappings) mappings.emplace_back(m.has_value() ? ToIndex(*m, material_count) : std::nullopt);

        auto &flags = layout.AttributeFlags[primitive_index];
        if (IsTriangleType(primitive.type)) {
            const auto prev_face_count = triangles.FaceCount();
            if (auto appended = AppendPrimitive(asset, primitive, triangles, triangle_attrs, deform, morph, triangle_vertex_count, flags); !appended) return std::unexpected{std::move(appended.error())};
            face_primitives.insert(face_primitives.end(), triangles.FaceCount() - prev_face_count, primitive_index);
            continue;
        }
        // Point and line shading keys off NORMAL and TANGENT, which the triangle append path records for itself.
        if (primitive.findAttribute("NORMAL") != primitive.attributes.end()) flags |= MeshAttributeBit_Normal;
        if (primitive.findAttribute("TANGENT") != primitive.attributes.end()) flags |= MeshAttributeBit_Tangent;
        const bool is_points = primitive.type == fastgltf::PrimitiveType::Points;
        auto &data = is_points ? points : lines;
        auto &element_primitives = is_points ? point_primitives : line_primitives;
        const auto prev_vertex_count = data.Positions.size();
        AppendNonTrianglePrimitive(asset, primitive, data, is_points ? point_attrs : line_attrs);
        element_primitives.insert(element_primitives.end(), data.Positions.size() - prev_vertex_count, primitive_index);
    }
    if (morph) {
        morph->DefaultWeights.assign(morph->TargetCount, 0.f);
        std::copy_n(source_mesh.weights.begin(), std::min(source_mesh.weights.size(), size_t(morph->TargetCount)), morph->DefaultWeights.begin());
    }

    const auto make_source = [&](::MeshData &data, ::MeshVertexAttributes &attrs, std::vector<uint32_t> &element_primitives) {
        return MeshSource{.Data = std::move(data), .Attrs = std::move(attrs), .Primitives = {std::move(element_primitives), layout.DefaultMaterials, layout.AttributeFlags}};
    };
    SourceMesh out;
    if (!triangles.Positions.empty() && triangles.FaceCount() > 0) {
        // Primitives without NORMAL are flat-shaded per the glTF spec.
        const bool any_normals = std::ranges::any_of(layout.AttributeFlags, [](uint32_t f) { return (f & MeshAttributeBit_Normal) != 0; });
        out.Triangles = make_source(triangles, triangle_attrs, face_primitives);
        out.Triangles->Deform = std::move(deform);
        out.Triangles->Morph = std::move(morph);
        out.Triangles->Weld = true;
        out.Triangles->FlatShaded = !any_normals;
    }
    if (!lines.Positions.empty()) out.Lines = make_source(lines, line_attrs, line_primitives);
    if (!points.Positions.empty()) out.Points = make_source(points, point_attrs, point_primitives);
    out.Layout = std::move(layout);
    return out;
}

std::vector<std::optional<uint32_t>> BuildNodeParentTable(const fastgltf::Asset &asset) {
    std::vector<std::optional<uint32_t>> parents(asset.nodes.size(), std::nullopt);
    for (uint32_t parent = 0; parent < asset.nodes.size(); ++parent) {
        for (const auto child_idx : asset.nodes[parent].children) {
            const auto child = ToIndex(child_idx, asset.nodes.size());
            if (child && !parents[*child]) parents[*child] = parent;
        }
    }
    return parents;
}

struct SceneTraversalData {
    std::vector<bool> InScene;
    std::vector<mat4> WorldTransforms;
};

SceneTraversalData TraverseSceneNodes(const fastgltf::Asset &asset, const std::vector<Transform> &local_transforms, uint32_t scene_index) {
    const auto nodes_count = asset.nodes.size();
    SceneTraversalData traversal{.InScene = std::vector(nodes_count, false), .WorldTransforms = std::vector(nodes_count, I4)};
    if (scene_index >= asset.scenes.size()) return traversal;

    const auto &scene = asset.scenes[scene_index];
    const auto traverse =
        [&](uint32_t node_index, const mat4 &parent_world, const auto &self) {
            if (node_index >= nodes_count) return;

            const auto &node = asset.nodes[node_index];
            const auto world = parent_world * ToMatrix(local_transforms[node_index]);
            traversal.InScene[node_index] = true;
            traversal.WorldTransforms[node_index] = world;
            for (const auto child_idx : node.children) {
                if (const auto child = ToIndex(child_idx, nodes_count)) self(*child, world, self);
            }
        };
    for (const auto root_idx : scene.nodeIndices) {
        if (const auto root = ToIndex(root_idx, nodes_count)) traverse(*root, I4, traverse);
    }

    return traversal;
}

std::optional<uint32_t> FindNearestMarkedAncestor(uint32_t node_index, const std::vector<std::optional<uint32_t>> &parents, const std::vector<bool> &marked) {
    auto parent = parents[node_index];
    while (parent) {
        if (marked[*parent]) return parent;
        parent = parents[*parent];
    }
    return {};
}

// Append a non-empty clip onto an entity's Anim component, creating the component on first use.
template<typename Anim, typename Clip>
bool AppendClip(state::Scene &r, state::Entity e, Clip &&clip) {
    if (clip.Channels.empty()) return false;
    if (auto *existing = r.try_edit<Anim>(e)) existing->Clips.emplace_back(std::forward<Clip>(clip));
    else r.emplace<Anim>(e, Anim{.Clips = {std::forward<Clip>(clip)}});
    return true;
}

std::optional<uint32_t> ComputeCommonAncestor(const std::vector<uint32_t> &nodes, const std::vector<std::optional<uint32_t>> &parents) {
    if (nodes.empty()) return {};

    const auto build_root_path = [&](uint32_t node_index) {
        std::vector<uint32_t> path;
        path.emplace_back(node_index);
        while (parents[node_index]) {
            node_index = *parents[node_index];
            path.emplace_back(node_index);
        }
        std::reverse(path.begin(), path.end());
        return path;
    };

    auto common_path = build_root_path(nodes.front());
    for (uint32_t i = 1; i < nodes.size() && !common_path.empty(); ++i) {
        const auto path = build_root_path(nodes[i]);
        const auto common_count = std::min(common_path.size(), path.size());
        size_t prefix = 0;
        while (prefix < common_count && common_path[prefix] == path[prefix]) ++prefix;
        common_path.resize(prefix);
    }

    if (common_path.empty()) return {};
    return common_path.back();
}

std::expected<Transform, std::string> ComputeJointRestLocal(
    uint32_t armature_index,
    uint32_t joint_node_index,
    std::optional<uint32_t> parent_joint_node_index,
    std::optional<uint32_t> anchor_node_index,
    const std::vector<std::optional<uint32_t>> &parents,
    const std::vector<Transform> &local_transforms
) {
    const auto rebased_parent_node_index = parent_joint_node_index ? parent_joint_node_index : anchor_node_index;
    if (!rebased_parent_node_index) return local_transforms[joint_node_index];

    mat4 rebased_local{I4};
    auto current = joint_node_index;
    while (current != *rebased_parent_node_index) {
        rebased_local = ToMatrix(local_transforms[current]) * rebased_local;
        if (!parents[current]) {
            return std::unexpected{std::format("glTF armature {} bone node {} cannot be rebased to ancestor node {}.", armature_index, joint_node_index, *rebased_parent_node_index)};
        }
        current = *parents[current];
    }
    return ToTransform(rebased_local);
}

std::expected<std::vector<uint32_t>, std::string> BuildParentBeforeChildJointOrder(const std::vector<uint32_t> &source_joint_nodes, const std::unordered_map<uint32_t, std::optional<uint32_t>> &joint_parent_map, uint32_t armature_index) {
    std::vector<uint32_t> ordered;
    ordered.reserve(source_joint_nodes.size());

    std::unordered_map<uint32_t, uint8_t> state;
    state.reserve(source_joint_nodes.size());

    const auto emit_joint = [&](uint32_t joint_node_index, const auto &self) -> std::expected<void, std::string> {
        const auto current_state = state[joint_node_index];
        if (current_state == 2) return {};
        if (current_state == 1) return std::unexpected{std::format("glTF armature {} has a cycle in bone ancestry at node {}.", armature_index, joint_node_index)};

        state[joint_node_index] = 1;
        if (const auto it = joint_parent_map.find(joint_node_index);
            it != joint_parent_map.end() && it->second) {
            if (auto parent_result = self(*it->second, self); !parent_result) return parent_result;
        }

        state[joint_node_index] = 2;
        ordered.emplace_back(joint_node_index);
        return {};
    };

    for (const auto joint_node_index : source_joint_nodes) {
        if (auto result = emit_joint(joint_node_index, emit_joint); !result) return std::unexpected{std::move(result.error())};
    }

    return ordered;
}

std::vector<mat4> LoadInverseBindMatrices(const fastgltf::Asset &asset, const fastgltf::Skin &skin, uint32_t joint_count) {
    std::vector<mat4> inverse_bind_matrices(joint_count, I4);
    if (!skin.inverseBindMatrices || *skin.inverseBindMatrices >= asset.accessors.size()) return inverse_bind_matrices;

    const auto &accessor = asset.accessors[*skin.inverseBindMatrices];
    if (accessor.type != fastgltf::AccessorType::Mat4 || accessor.count == 0) return inverse_bind_matrices;
    if (accessor.count <= joint_count) {
        fastgltf::copyFromAccessor<mat4>(asset, accessor, inverse_bind_matrices.data());
    } else {
        std::vector<mat4> ibm(accessor.count);
        fastgltf::copyFromAccessor<mat4>(asset, accessor, ibm.data());
        std::copy_n(ibm.begin(), joint_count, inverse_bind_matrices.begin());
    }
    return inverse_bind_matrices;
}

std::string MakeNodeName(const fastgltf::Asset &asset, uint32_t node_index, std::optional<uint32_t> source_mesh_index = {}) {
    const auto &node = asset.nodes[node_index];
    if (!node.name.empty()) return std::string{node.name};

    if (source_mesh_index && *source_mesh_index < asset.meshes.size()) {
        const auto &mesh_name = asset.meshes[*source_mesh_index].name;
        if (!mesh_name.empty()) return std::string{mesh_name};
    }

    return std::format("Node{}", node_index);
}
std::vector<Transform> ReadInstanceTransforms(const fastgltf::Asset &asset, const fastgltf::Node &node) {
    if (node.instancingAttributes.empty()) return {};

    const auto t_attr = node.findInstancingAttribute("TRANSLATION");
    const auto r_attr = node.findInstancingAttribute("ROTATION");
    const auto s_attr = node.findInstancingAttribute("SCALE");

    const uint32_t instance_count = t_attr != node.instancingAttributes.end() ? asset.accessors[t_attr->accessorIndex].count :
        r_attr != node.instancingAttributes.end()                             ? asset.accessors[r_attr->accessorIndex].count :
        s_attr != node.instancingAttributes.end()                             ? asset.accessors[s_attr->accessorIndex].count :
                                                                                0;
    if (instance_count == 0) return {};

    std::vector<Transform> transforms(instance_count);
    if (t_attr != node.instancingAttributes.end()) {
        const auto &accessor = asset.accessors[t_attr->accessorIndex];
        fastgltf::iterateAccessorWithIndex<vec3>(asset, accessor, [&](const vec3 &v, auto i) { transforms[i].P = v; });
    }
    if (r_attr != node.instancingAttributes.end()) {
        const auto &accessor = asset.accessors[r_attr->accessorIndex];
        fastgltf::iterateAccessorWithIndex<vec4>(asset, accessor, [&](const vec4 &v, auto i) {
            transforms[i].R = numeric::Normalize(std::bit_cast<quat>(v));
        });
    }
    if (s_attr != node.instancingAttributes.end()) {
        const auto &accessor = asset.accessors[s_attr->accessorIndex];
        fastgltf::iterateAccessorWithIndex<vec3>(asset, accessor, [&](const vec3 &v, auto i) { transforms[i].S = v; });
    }

    return transforms;
}

::Camera ConvertCamera(const fastgltf::Camera &cam) {
    return std::visit(
        [](const auto &source) -> ::Camera {
            using P = std::decay_t<decltype(source)>;
            if constexpr (std::is_same_v<P, fastgltf::Camera::Perspective>) {
                return Perspective{.FieldOfViewRad = source.yfov, .FarClip = source.zfar, .NearClip = source.znear, .AspectRatio = source.aspectRatio};
            } else {
                return Orthographic{.Mag = {source.xmag, source.ymag}, .FarClip = source.zfar, .NearClip = source.znear};
            }
        },
        cam.camera
    );
}

PunctualLight ConvertLight(const fastgltf::Light &light) {
    PunctualLight pl{
        .Range = 0.f,
        .Color = ToVec3(light.color),
        .Intensity = light.intensity,
        .InnerConeCos = 0.f,
        .OuterConeCos = 0.f,
        .Type = PunctualLightType::Point,
    };
    switch (light.type) {
        case fastgltf::LightType::Directional:
            pl.Type = PunctualLightType::Directional;
            break;
        case fastgltf::LightType::Point:
            pl.Type = PunctualLightType::Point;
            pl.Range = light.range ? *light.range : 0.f;
            break;
        case fastgltf::LightType::Spot: {
            const auto outer = light.outerConeAngle ? *light.outerConeAngle : std::numbers::pi_v<float> / 4.f;
            pl.Type = PunctualLightType::Spot;
            pl.Range = light.range ? *light.range : 0.f;
            pl.InnerConeCos = std::cos(std::clamp(light.innerConeAngle ? *light.innerConeAngle : 0.f, 0.f, outer));
            pl.OuterConeCos = std::cos(outer);
            break;
        }
    }
    return pl;
}

std::optional<ImageBasedLight> ConvertIBL(const fastgltf::Asset &asset, size_t scene_index) {
    const auto ibl_idx = ToIndex(asset.scenes[scene_index].imageBasedLightIndex, asset.imageBasedLights.size());
    if (!ibl_idx) return std::nullopt;
    const auto &src_ibl = asset.imageBasedLights[*ibl_idx];
    ImageBasedLight ibl{
        .Rotation = numeric::Normalize(std::bit_cast<quat>(src_ibl.rotation)),
        .SpecularImageSize = src_ibl.specularImageSize,
        .Intensity = std::max(0.f, src_ibl.intensity),
        .Name = src_ibl.name.empty() ? std::format("ImageBasedLight{}", *ibl_idx) : std::string{src_ibl.name},
    };
    ibl.SpecularImageIndicesByMip.reserve(src_ibl.specularImages.size());
    for (const auto &mip : src_ibl.specularImages) {
        std::array<uint32_t, 6> faces{};
        for (size_t face = 0; face < 6; ++face) faces[face] = mip[face];
        ibl.SpecularImageIndicesByMip.emplace_back(faces);
    }
    if (src_ibl.irradianceCoefficients) {
        std::array<vec3, 9> coefficients{};
        for (size_t i = 0; i < 9; ++i) {
            coefficients[i] = std::bit_cast<vec3>((*src_ibl.irradianceCoefficients)[i]);
        }
        ibl.IrradianceCoefficients = coefficients;
    }
    return ibl;
}

state::Entity ActiveSceneEntity(const state::Scene &r) {
    for (const auto e : r.view<const ActiveScene>()) return e;
    return state::Null;
}
// No SceneMembership (single-scene) means the node is in the sole scene, so always in the active one.
bool EntityInActiveScene(const state::Scene &r, state::Entity active_scene, state::Entity e) {
    const auto *sm = r.try_get<const SceneMembership>(e);
    return !sm || std::ranges::find(sm->Scenes, active_scene) != sm->Scenes.end();
}

// Toggle RenderInstance so only nodes in the active scene render. No-op for single-scene assets.
void ApplySceneVisibility(state::Scene &r) {
    const auto active = ActiveSceneEntity(r);
    for (auto [e, sm, _i] : r.view<const SceneMembership, const Instance>().each()) {
        if (std::ranges::find(sm.Scenes, active) != sm.Scenes.end()) Show(r, e);
        else Hide(r, e);
    }
}

// Selects an active imported entity by source order and camera, mesh, armature, root-empty, then object priority.
void ApplyActiveSceneSelection(state::Scene &r) {
    const auto active_scene = ActiveSceneEntity(r);

    // Armatures sort after source-indexed objects.
    std::vector<std::pair<uint32_t, state::Entity>> ordered;
    for (const auto [e, node, _] : r.view<const GltfNode, const ObjectKind>().each()) {
        if (EntityInActiveScene(r, active_scene, e)) ordered.emplace_back(node.Index.value_or(std::numeric_limits<uint32_t>::max()), e);
    }
    std::ranges::sort(ordered);

    const auto priority = [&](state::Entity e) {
        switch (r.get<const ObjectKind>(e).Value) {
            case ObjectType::Camera: return 0;
            case ObjectType::Mesh: return 1;
            case ObjectType::Armature: return 2;
            case ObjectType::Empty: return r.get<const GltfNode>(e).Parent ? 4 : 3;
            default: return 4;
        }
    };
    state::Entity active = state::Null;
    int best = std::numeric_limits<int>::max();
    for (const auto &[_, e] : ordered) {
        if (const auto p = priority(e); p < best) {
            best = p;
            active = e;
        }
    }

    r.clear<Active, Selected>();
    if (active != state::Null) r.emplace<Active>(active);
    for (const auto &[_, e] : ordered) r.emplace<Selected>(e);
}
// Header, samplers, images, textures, required extensions, and the default scene's IBL.
// Image bytes move into the project store when one is open, and an image with a file to reload from keeps no bytes.
std::expected<SourceAssets, std::string> ReadSourceAssets(state::Scene &r, const fastgltf::Asset &asset, const std::filesystem::path &stored_path, ExtrasMap &&extras, uint32_t scene_index) {
    SourceAssets sa{
        .Copyright = asset.assetInfo ? std::string{asset.assetInfo->copyright} : std::string{},
        .Generator = asset.assetInfo ? std::string{asset.assetInfo->generator} : std::string{},
        .MinVersion = asset.assetInfo ? std::string{asset.assetInfo->minVersion} : std::string{},
        .AssetExtras = asset.assetInfo ? std::string{asset.assetInfo->extras} : std::string{},
        .AssetExtensions = asset.assetInfo ? std::string{asset.assetInfo->extensions} : std::string{},
        .ExtensionsRequired = {},
        .ExtrasByEntity = std::move(extras),
        .MaterialMetas = {},
        .Textures = {},
        .Images = {},
        .Samplers = {},
        .AnimationOrder = {},
        .ImageBasedLight = ConvertIBL(asset, scene_index),
    };
    sa.ExtensionsRequired.reserve(asset.extensionsRequired.size());
    for (const auto &e : asset.extensionsRequired) sa.ExtensionsRequired.emplace_back(e);
    sa.Samplers.reserve(asset.samplers.size());
    for (const auto &sampler : asset.samplers) {
        sa.Samplers.emplace_back(Sampler{
            .MagFilter = ToFilter(sampler.magFilter),
            .MinFilter = ToFilter(sampler.minFilter),
            .WrapS = ToWrap(sampler.wrapS),
            .WrapT = ToWrap(sampler.wrapT),
            .Name = std::string{sampler.name},
        });
    }
    sa.Images.reserve(asset.images.size());
    const auto source_dir = AbsoluteScenePath(stored_path).parent_path();
    for (uint32_t image_index = 0; image_index < asset.images.size(); ++image_index) {
        auto image = ReadImage(asset, image_index, source_dir);
        if (!image) return std::unexpected{std::move(image.error())};
        if (!image->SourcePath.empty()) image->SourcePath = project::AssetReference(r, image->SourcePath).string();
        if (auto *files = r.ctx().find<project::Assets>(); files && !project::Assets::IsReference(image->SourcePath)) {
            const auto stored = files->Store("image.bin", image->Bytes);
            if (!stored) return std::unexpected{stored.error()};
            image->SourcePath = stored->string();
        }
        if (!image->SourcePath.empty()) image->Bytes = {};
        sa.Images.emplace_back(std::move(*image));
    }
    sa.Textures.reserve(asset.textures.size());
    for (const auto &texture : asset.textures) {
        sa.Textures.emplace_back(Texture{
            .SamplerIndex = ToIndex(texture.samplerIndex, asset.samplers.size()),
            .ImageIndex = ToIndex(texture.imageIndex, asset.images.size()),
            .WebpImageIndex = ToIndex(texture.webpImageIndex, asset.images.size()),
            .BasisuImageIndex = ToIndex(texture.basisuImageIndex, asset.images.size()),
            .DdsImageIndex = ToIndex(texture.ddsImageIndex, asset.images.size()),
            .Name = std::string{texture.name},
        });
    }
    return sa;
}

// Render materials and source metadata in parallel, with a trailing implicit default for primitives without a material.
// Texture slots hold glTF texture indices until ImportMaterials maps them to bindless slots.
struct SourceMaterials {
    std::vector<PBRMaterial> Materials;
    std::vector<MaterialSourceMeta> Metas;
};

SourceMaterials ReadMaterials(const fastgltf::Asset &asset) {
    SourceMaterials out;
    out.Materials.reserve(asset.materials.size() + 1u);
    out.Metas.reserve(asset.materials.size() + 1u);
    for (uint32_t material_index = 0; material_index < asset.materials.size(); ++material_index) {
        const auto &material = asset.materials[material_index];
        using M = MaterialSourceMeta;
        MaterialSourceMeta meta;
        meta.NameWasEmpty = material.name.empty();
        PBRMaterial pbr{
            .BaseColorFactor = ToVec4(material.pbrData.baseColorFactor),
            .EmissiveFactor = ToVec3(material.emissiveFactor),
            .MetallicFactor = material.pbrData.metallicFactor,
            .RoughnessFactor = material.pbrData.roughnessFactor,
            .NormalScale = material.normalTexture ? material.normalTexture->scale : 1.f,
            .OcclusionStrength = material.occlusionTexture ? material.occlusionTexture->strength : 1.f,
            .AlphaMode = ToAlphaMode(material.alphaMode),
            .AlphaCutoff = material.alphaCutoff,
            .DoubleSided = material.doubleSided ? 1u : 0u,
            .Unlit = material.unlit ? 1u : 0u,
            .BaseColorTexture = ToTextureIndex(material.pbrData.baseColorTexture, asset, &meta.BaseSlotMeta[0]),
            .MetallicRoughnessTexture = ToTextureIndex(material.pbrData.metallicRoughnessTexture, asset, &meta.BaseSlotMeta[1]),
            .NormalTexture = ToTextureIndex(material.normalTexture, asset, &meta.BaseSlotMeta[2]),
            .OcclusionTexture = ToTextureIndex(material.occlusionTexture, asset, &meta.BaseSlotMeta[3]),
            .EmissiveTexture = ToTextureIndex(material.emissiveTexture, asset, &meta.BaseSlotMeta[4]),
        };
        if (material.ior) {
            pbr.Ior = *material.ior;
            meta.ExtensionPresence |= M::ExtIor;
        }
        if (material.dispersion) {
            pbr.Dispersion = *material.dispersion;
            meta.ExtensionPresence |= M::ExtDispersion;
        }
        if (material.emissiveStrength) {
            const float s = *material.emissiveStrength;
            meta.EmissiveStrength = s;
            meta.ExtensionPresence |= M::ExtEmissiveStrength;
            pbr.EmissiveFactor *= s; // collapse strength into factor for GPU
        }

        if (material.sheen) {
            meta.ExtensionPresence |= M::ExtSheen;
            pbr.Sheen = ::Sheen{
                .ColorFactor = ToVec3(material.sheen->sheenColorFactor),
                .RoughnessFactor = material.sheen->sheenRoughnessFactor,
                .ColorTexture = ToTextureIndex(material.sheen->sheenColorTexture, asset),
                .RoughnessTexture = ToTextureIndex(material.sheen->sheenRoughnessTexture, asset),
            };
        }
        if (material.specular) {
            meta.ExtensionPresence |= M::ExtSpecular;
            pbr.Specular = ::Specular{
                .Factor = material.specular->specularFactor,
                .ColorFactor = ToVec3(material.specular->specularColorFactor),
                .Texture = ToTextureIndex(material.specular->specularTexture, asset),
                .ColorTexture = ToTextureIndex(material.specular->specularColorTexture, asset),
            };
        }
        if (material.transmission) {
            meta.ExtensionPresence |= M::ExtTransmission;
            pbr.Transmission = ::Transmission{
                .Factor = material.transmission->transmissionFactor,
                .Texture = ToTextureIndex(material.transmission->transmissionTexture, asset),
            };
        }
        if (material.diffuseTransmission) {
            meta.ExtensionPresence |= M::ExtDiffuseTransmission;
            pbr.DiffuseTransmission = ::DiffuseTransmission{
                .Factor = material.diffuseTransmission->diffuseTransmissionFactor,
                .ColorFactor = ToVec3(material.diffuseTransmission->diffuseTransmissionColorFactor),
                .Texture = ToTextureIndex(material.diffuseTransmission->diffuseTransmissionTexture, asset),
                .ColorTexture = ToTextureIndex(material.diffuseTransmission->diffuseTransmissionColorTexture, asset),
            };
        }
        if (material.volume) {
            meta.ExtensionPresence |= M::ExtVolume;
            const float ad = material.volume->attenuationDistance;
            pbr.Volume = ::Volume{
                .ThicknessFactor = material.volume->thicknessFactor,
                .AttenuationColor = ToVec3(material.volume->attenuationColor),
                .AttenuationDistance = (std::isinf(ad) || ad <= 0.f) ? 0.f : ad,
                .ThicknessTexture = ToTextureIndex(material.volume->thicknessTexture, asset),
            };
        }
        if (material.clearcoat) {
            meta.ExtensionPresence |= M::ExtClearcoat;
            pbr.Clearcoat = ::Clearcoat{
                .Factor = material.clearcoat->clearcoatFactor,
                .RoughnessFactor = material.clearcoat->clearcoatRoughnessFactor,
                .NormalScale = material.clearcoat->clearcoatNormalTexture ? material.clearcoat->clearcoatNormalTexture->scale : 1.f,
                .Texture = ToTextureIndex(material.clearcoat->clearcoatTexture, asset),
                .RoughnessTexture = ToTextureIndex(material.clearcoat->clearcoatRoughnessTexture, asset),
                .NormalTexture = ToTextureIndex(material.clearcoat->clearcoatNormalTexture, asset),
            };
        }
        if (material.anisotropy) {
            meta.ExtensionPresence |= M::ExtAnisotropy;
            pbr.Anisotropy = ::Anisotropy{
                .Strength = material.anisotropy->anisotropyStrength,
                .Rotation = material.anisotropy->anisotropyRotation,
                .Texture = ToTextureIndex(material.anisotropy->anisotropyTexture, asset),
            };
        }
        if (material.iridescence) {
            meta.ExtensionPresence |= M::ExtIridescence;
            pbr.Iridescence = ::Iridescence{
                .Factor = material.iridescence->iridescenceFactor,
                .Ior = material.iridescence->iridescenceIor,
                .ThicknessMinimum = material.iridescence->iridescenceThicknessMinimum,
                .ThicknessMaximum = material.iridescence->iridescenceThicknessMaximum,
                .Texture = ToTextureIndex(material.iridescence->iridescenceTexture, asset),
                .ThicknessTexture = ToTextureIndex(material.iridescence->iridescenceThicknessTexture, asset),
            };
        }

        for (uint32_t s = 0; s < MTS_Count; ++s) meta.TextureSlots[s] = MaterialTextureSlots[s].Get(pbr).Slot;
        out.Metas.emplace_back(std::move(meta));
        out.Materials.emplace_back(std::move(pbr));
    }
    out.Materials.emplace_back();
    out.Metas.emplace_back(MaterialSourceMeta{.ImplicitDefault = true});
    return out;
}

// Node facts derived from the document before any scene mutation.
struct NodePlan {
    std::vector<std::optional<uint32_t>> Parents;
    std::vector<Transform> LocalTransforms;
    std::vector<std::optional<mat4>> SourceMatrices;
    // Merged over every scene with the default scene first, so a node shared between scenes keeps the default scene's placement.
    std::vector<bool> InScene;
    std::vector<mat4> WorldTransforms;
    std::vector<uint32_t> SceneMask;
    std::vector<bool> UsedSkin, IsBone;
    // Bone nodes render only when they also carry mesh data.
    std::vector<bool> IsObjectEmitted;
    std::vector<std::optional<uint32_t>> NearestObjectAncestor;
    // Armature-root node per skin. Empty means the scene root.
    std::vector<std::optional<uint32_t>> SkinArmaNode;
    // Valid joint nodes per used skin, deduped, in source order.
    std::vector<std::vector<uint32_t>> SkinJointNodes;
};

NodePlan PlanNodes(const fastgltf::Asset &asset, uint32_t scene_index) {
    const auto node_count = asset.nodes.size();
    NodePlan plan{
        .Parents = BuildNodeParentTable(asset),
        .LocalTransforms = std::vector<Transform>(node_count),
        .SourceMatrices = std::vector<std::optional<mat4>>(node_count),
        .InScene = std::vector(node_count, false),
        .WorldTransforms = std::vector(node_count, I4),
        .SceneMask = std::vector<uint32_t>(node_count, 0u),
        .UsedSkin = std::vector(asset.skins.size(), false),
        .IsBone = std::vector(node_count, false),
        .IsObjectEmitted = std::vector(node_count, false),
        .NearestObjectAncestor = std::vector<std::optional<uint32_t>>(node_count),
        .SkinArmaNode = std::vector<std::optional<uint32_t>>(asset.skins.size()),
        .SkinJointNodes = std::vector<std::vector<uint32_t>>(asset.skins.size()),
    };
    for (uint32_t node_index = 0; node_index < node_count; ++node_index) {
        const auto &fg_transform = asset.nodes[node_index].transform;
        if (std::holds_alternative<fastgltf::TRS>(fg_transform)) {
            plan.LocalTransforms[node_index] = TrsToTransform(std::get<fastgltf::TRS>(fg_transform));
        } else {
            const auto &fm = std::get<fastgltf::math::fmat4x4>(fg_transform);
            plan.SourceMatrices[node_index] = std::bit_cast<mat4>(fm);

            fastgltf::math::fvec3 scale, translation;
            fastgltf::math::fquat rotation;
            fastgltf::math::decomposeTransformMatrix(fm, scale, rotation, translation);
            plan.LocalTransforms[node_index] = Transform{ToVec3(translation), numeric::Normalize(ToQuat(rotation)), ToVec3(scale)};
        }
    }
    const auto merge_scene = [&](uint32_t si) {
        const auto t = TraverseSceneNodes(asset, plan.LocalTransforms, si);
        for (uint32_t i = 0; i < node_count; ++i) {
            if (!t.InScene[i]) continue;
            plan.SceneMask[i] |= (1u << si);
            if (!plan.InScene[i]) {
                plan.InScene[i] = true;
                plan.WorldTransforms[i] = t.WorldTransforms[i];
            }
        }
    };
    merge_scene(scene_index);
    for (uint32_t s = 0; s < asset.scenes.size(); ++s) {
        if (s != scene_index) merge_scene(s);
    }

    for (uint32_t node_index = 0; node_index < node_count; ++node_index) {
        if (!plan.InScene[node_index]) continue;
        if (const auto skin_index = ToIndex(asset.nodes[node_index].skinIndex, asset.skins.size())) plan.UsedSkin[*skin_index] = true;
    }

    // A skin's armature root is the nearest non-bone ancestor of its joints' common ancestor.
    // Every payload-free node between a joint and that root also becomes a bone.
    std::vector<bool> node_carries_payload(node_count, false);
    for (uint32_t node_index = 0; node_index < node_count; ++node_index) {
        const auto &node = asset.nodes[node_index];
        node_carries_payload[node_index] = ToIndex(node.meshIndex, asset.meshes.size()).has_value() ||
            ToIndex(node.cameraIndex, asset.cameras.size()).has_value() ||
            ToIndex(node.lightIndex, asset.lights.size()).has_value() ||
            bool(node.physicsRigidBody) || !node.instancingAttributes.empty();
    }
    std::vector<std::optional<uint32_t>> skin_lca(asset.skins.size());
    for (uint32_t skin_index = 0; skin_index < asset.skins.size(); ++skin_index) {
        if (!plan.UsedSkin[skin_index]) continue;
        const auto &skin = asset.skins[skin_index];
        auto &joint_nodes = plan.SkinJointNodes[skin_index];
        std::unordered_set<uint32_t> seen;
        for (const auto joint_idx : skin.joints) {
            if (const auto joint = ToIndex(joint_idx, node_count); joint && seen.emplace(*joint).second) joint_nodes.emplace_back(*joint);
        }
        if (joint_nodes.empty()) continue;
        auto lca_candidates = joint_nodes;
        if (const auto skel = ToIndex(skin.skeleton, node_count)) lca_candidates.emplace_back(*skel);
        skin_lca[skin_index] = ComputeCommonAncestor(lca_candidates, plan.Parents);
        for (const auto joint : joint_nodes) plan.IsBone[joint] = true;
    }
    for (bool changed = true; changed;) {
        changed = false;
        for (uint32_t skin_index = 0; skin_index < asset.skins.size(); ++skin_index) {
            if (plan.SkinJointNodes[skin_index].empty()) continue;
            auto arma = skin_lca[skin_index];
            while (arma && plan.IsBone[*arma]) arma = plan.Parents[*arma];
            plan.SkinArmaNode[skin_index] = arma;
            for (const auto joint : plan.SkinJointNodes[skin_index]) {
                for (std::optional<uint32_t> cur = joint; cur && cur != arma; cur = plan.Parents[*cur]) {
                    if (!plan.IsBone[*cur] && !node_carries_payload[*cur]) {
                        plan.IsBone[*cur] = true;
                        changed = true;
                    }
                }
            }
        }
    }

    for (uint32_t node_index = 0; node_index < node_count; ++node_index) {
        const bool has_mesh = ToIndex(asset.nodes[node_index].meshIndex, asset.meshes.size()).has_value();
        plan.IsObjectEmitted[node_index] = plan.InScene[node_index] && (has_mesh || !plan.IsBone[node_index]);
    }
    for (uint32_t node_index = 0; node_index < node_count; ++node_index) {
        plan.NearestObjectAncestor[node_index] = FindNearestMarkedAncestor(node_index, plan.Parents, plan.IsObjectEmitted);
    }
    return plan;
}

// One armature per distinct armature root, consuming every skin anchored there, with its bones validated and rest-posed.
// A bone's pose world then composes the same node transforms as the spec's global joint transform.
struct ArmaturePlan {
    std::optional<uint32_t> ArmaNode;
    std::vector<uint32_t> SkinIndices;
    // Parent before child, with each bone's parent bone and rest transform at the same position.
    std::vector<uint32_t> BoneNodes;
    std::vector<std::optional<uint32_t>> BoneParents;
    std::vector<Transform> RestLocals;
};

std::expected<std::vector<ArmaturePlan>, std::string> PlanArmatures(const fastgltf::Asset &asset, const NodePlan &plan, std::span<const SourceMesh> source_meshes, const std::filesystem::path &source_path) {
    std::vector<ArmaturePlan> groups;
    for (uint32_t skin_index = 0; skin_index < asset.skins.size(); ++skin_index) {
        if (!plan.UsedSkin[skin_index] || plan.SkinJointNodes[skin_index].empty()) continue;
        auto it = std::ranges::find(groups, plan.SkinArmaNode[skin_index], &ArmaturePlan::ArmaNode);
        if (it == groups.end()) it = groups.emplace(groups.end(), ArmaturePlan{.ArmaNode = plan.SkinArmaNode[skin_index], .SkinIndices = {}, .BoneNodes = {}, .BoneParents = {}, .RestLocals = {}});
        it->SkinIndices.emplace_back(skin_index);
    }
    // A skin binds only through an emitted mesh instance that references it.
    std::vector<bool> skin_has_instance(asset.skins.size(), false);
    for (uint32_t node_index = 0; node_index < asset.nodes.size(); ++node_index) {
        if (!plan.IsObjectEmitted[node_index]) continue;
        const auto &node = asset.nodes[node_index];
        const auto mesh_index = ToIndex(node.meshIndex, asset.meshes.size());
        if (const auto skin_index = ToIndex(node.skinIndex, asset.skins.size()); skin_index && mesh_index && source_meshes[*mesh_index].HasParts()) skin_has_instance[*skin_index] = true;
    }
    for (uint32_t group_index = 0; group_index < groups.size(); ++group_index) {
        auto &group = groups[group_index];
        const auto arma_node = group.ArmaNode;
        if (arma_node && !plan.InScene[*arma_node]) {
            return std::unexpected{std::format("glTF import failed for '{}': skin {} armature root node {} is not in the imported scene.", source_path.string(), group.SkinIndices.front(), *arma_node)};
        }
        for (const auto skin_index : group.SkinIndices) {
            if (!skin_has_instance[skin_index]) return std::unexpected{std::format("glTF import failed '{}': skin {} is used but no mesh instances were emitted for skin binding.", source_path.string(), skin_index)};
        }

        // Bone nodes: every bone node on a path from a joint up to the root (exclusive), first-seen order.
        std::vector<uint32_t> source_bone_nodes;
        std::vector<bool> in_group(asset.nodes.size(), false);
        for (const auto skin_index : group.SkinIndices) {
            for (const auto joint : plan.SkinJointNodes[skin_index]) {
                for (std::optional<uint32_t> cur = joint; cur && cur != arma_node; cur = plan.Parents[*cur]) {
                    if (!plan.IsBone[*cur]) continue;
                    if (in_group[*cur]) break;
                    in_group[*cur] = true;
                    source_bone_nodes.emplace_back(*cur);
                }
            }
        }
        std::unordered_map<uint32_t, std::optional<uint32_t>> bone_parent_map;
        bone_parent_map.reserve(source_bone_nodes.size());
        for (const auto node : source_bone_nodes) bone_parent_map.emplace(node, FindNearestMarkedAncestor(node, plan.Parents, in_group));

        auto ordered = BuildParentBeforeChildJointOrder(source_bone_nodes, bone_parent_map, group_index);
        if (!ordered) return std::unexpected{std::move(ordered.error())};
        group.BoneNodes = std::move(*ordered);
        group.BoneParents.reserve(group.BoneNodes.size());
        group.RestLocals.reserve(group.BoneNodes.size());
        for (const auto node : group.BoneNodes) {
            const auto parent_node = bone_parent_map.at(node);
            auto rest_local = ComputeJointRestLocal(group_index, node, parent_node, arma_node, plan.Parents, plan.LocalTransforms);
            if (!rest_local) return std::unexpected{std::move(rest_local.error())};
            group.BoneParents.emplace_back(parent_node);
            group.RestLocals.emplace_back(*rest_local);
        }
    }
    return groups;
}

// Commit phase. Everything below runs after validation and cannot fail.

struct PhysicsResources {
    std::vector<state::Entity> Materials, JointDefs;
};

// KHR_physics_rigid_bodies document resources. Collision filters follow in ImportNodePhysics with the node colliders that reference them.
PhysicsResources ImportPhysicsResources(state::Scene &r, const fastgltf::Asset &asset) {
    PhysicsResources out;
    out.Materials.reserve(asset.physicsMaterials.size());
    for (uint32_t i = 0; i < asset.physicsMaterials.size(); ++i) {
        const auto &src = asset.physicsMaterials[i];
        const auto e = r.create();
        r.emplace<PhysicsMaterial>(e, PhysicsMaterial{.StaticFriction = src.staticFriction, .DynamicFriction = src.dynamicFriction, .Restitution = src.restitution, .FrictionCombine = ToCombineMode(src.frictionCombine), .RestitutionCombine = ToCombineMode(src.restitutionCombine)});
        r.emplace<SourceIndex>(e, i);
        out.Materials.emplace_back(e);
    }
    out.JointDefs.reserve(asset.physicsJoints.size());
    for (uint32_t i = 0; i < asset.physicsJoints.size(); ++i) {
        const auto &src = asset.physicsJoints[i];
        PhysicsJointDef def;
        for (const auto &lim : src.limits) {
            def.Limits.emplace_back(PhysicsJointLimit{
                .LinearAxes = {lim.linearAxes.begin(), lim.linearAxes.end()},
                .AngularAxes = {lim.angularAxes.begin(), lim.angularAxes.end()},
                .Min = lim.min ? std::optional{float(*lim.min)} : std::nullopt,
                .Max = lim.max ? std::optional{float(*lim.max)} : std::nullopt,
                .Stiffness = lim.stiffness ? std::optional{float(*lim.stiffness)} : std::nullopt,
                .Damping = float(lim.damping),
            });
        }
        for (const auto &drv : src.drives) {
            // fastgltf zero-initializes maxForce when absent, and the KHR default is FLT_MAX.
            def.Drives.emplace_back(PhysicsJointDrive{
                .Type = drv.type == fastgltf::DriveType::Angular ? PhysicsDriveType::Angular : PhysicsDriveType::Linear,
                .Mode = drv.mode == fastgltf::DriveMode::Acceleration ? PhysicsDriveMode::Acceleration : PhysicsDriveMode::Force,
                .Axis = drv.axis,
                .MaxForce = drv.maxForce > 0 ? float(drv.maxForce) : std::numeric_limits<float>::max(),
                .PositionTarget = float(drv.positionTarget),
                .VelocityTarget = float(drv.velocityTarget),
                .Stiffness = float(drv.stiffness),
                .Damping = float(drv.damping),
            });
        }
        const auto e = r.create();
        r.emplace<PhysicsJointDef>(e, std::move(def));
        r.emplace<SourceIndex>(e, i);
        out.JointDefs.emplace_back(e);
    }
    return out;
}

struct ImportedMaterials {
    // GPU material index per source material, the trailing entry being the implicit default.
    std::vector<uint32_t> IndexByGltfMaterial;
    std::vector<MaterializedTexture> Textures;
};

// Appends every source material to the GPU material buffer with its textures queued for upload and mapped to bindless slots.
ImportedMaterials ImportMaterials(state::Scene &r, const fastgltf::Asset &asset, const SourceAssets &sa, std::span<const PBRMaterial> source_materials) {
    auto &slots = r.ctx().get<mtl::BindlessSet>();
    auto &buffers = r.ctx().get<GpuBuffers>();
    auto &textures = r.ctx().get<TextureStore>();
    ImportedMaterials out;

    // Equivalent glTF textures share one TextureEntry, keyed by resolved image, sampler, and color space.
    std::unordered_map<uint64_t, uint32_t> texture_slot_cache;
    const auto resolve_texture_slot = [&](uint32_t texture_index, TextureColorSpace color_space) -> uint32_t {
        if (texture_index >= sa.Textures.size()) return InvalidSlot;
        const auto &src_texture = sa.Textures[texture_index];
        const auto image_index = ResolveImageIndex(src_texture);
        if (!image_index || *image_index >= sa.Images.size()) return InvalidSlot;

        const auto sampler_index = src_texture.SamplerIndex.value_or(InvalidSlot);
        const auto cache_key = (uint64_t(*image_index) << 33u) | (uint64_t(sampler_index) << 1u) | (color_space == TextureColorSpace::Srgb ? 1u : 0u);
        if (const auto it = texture_slot_cache.find(cache_key); it != texture_slot_cache.end()) return it->second;

        const auto *src_sampler = src_texture.SamplerIndex && *src_texture.SamplerIndex < sa.Samplers.size() ? &sa.Samplers[*src_texture.SamplerIndex] : nullptr;
        static constexpr auto ToSamplerAddressMode = [](Wrap wrap) {
            switch (wrap) {
                case Wrap::ClampToEdge: return MTL::SamplerAddressModeClampToEdge;
                case Wrap::MirroredRepeat: return MTL::SamplerAddressModeMirrorRepeat;
                case Wrap::Repeat: return MTL::SamplerAddressModeRepeat;
            }
            return MTL::SamplerAddressModeRepeat;
        };
        static constexpr auto ToSamplerConfig = [](const Sampler *sampler) -> SamplerConfig {
            if (!sampler) return {.MinFilter = MTL::SamplerMinMagFilterLinear, .MagFilter = MTL::SamplerMinMagFilterLinear, .MipmapMode = MTL::SamplerMipFilterLinear, .UsesMipmaps = true};

            const auto mag_filter = sampler->MagFilter && *sampler->MagFilter == Filter::Nearest ? MTL::SamplerMinMagFilterNearest : MTL::SamplerMinMagFilterLinear;
            switch (sampler->MinFilter.value_or(Filter::LinearMipMapLinear)) {
                case Filter::Nearest:
                    return {.MinFilter = MTL::SamplerMinMagFilterNearest, .MagFilter = mag_filter, .MipmapMode = MTL::SamplerMipFilterNearest, .UsesMipmaps = false};
                case Filter::Linear:
                    return {.MinFilter = MTL::SamplerMinMagFilterLinear, .MagFilter = mag_filter, .MipmapMode = MTL::SamplerMipFilterNearest, .UsesMipmaps = false};
                case Filter::NearestMipMapNearest:
                    return {.MinFilter = MTL::SamplerMinMagFilterNearest, .MagFilter = mag_filter, .MipmapMode = MTL::SamplerMipFilterNearest, .UsesMipmaps = true};
                case Filter::LinearMipMapNearest:
                    return {.MinFilter = MTL::SamplerMinMagFilterLinear, .MagFilter = mag_filter, .MipmapMode = MTL::SamplerMipFilterNearest, .UsesMipmaps = true};
                case Filter::NearestMipMapLinear:
                    return {.MinFilter = MTL::SamplerMinMagFilterNearest, .MagFilter = mag_filter, .MipmapMode = MTL::SamplerMipFilterLinear, .UsesMipmaps = true};
                case Filter::LinearMipMapLinear:
                    return {.MinFilter = MTL::SamplerMinMagFilterLinear, .MagFilter = mag_filter, .MipmapMode = MTL::SamplerMipFilterLinear, .UsesMipmaps = true};
            }
            return {.MinFilter = MTL::SamplerMinMagFilterLinear, .MagFilter = mag_filter, .MipmapMode = MTL::SamplerMipFilterLinear, .UsesMipmaps = true};
        };

        const auto sampler_slot = AllocateSamplerSlot(slots);
        const PendingTextureUpload upload{
            .SamplerSlot = sampler_slot,
            .Source = PendingTextureUpload::GltfImageRef{*image_index},
            .Params = {
                .ColorSpace = color_space,
                .WrapS = src_sampler ? ToSamplerAddressMode(src_sampler->WrapS) : MTL::SamplerAddressModeRepeat,
                .WrapT = src_sampler ? ToSamplerAddressMode(src_sampler->WrapT) : MTL::SamplerAddressModeRepeat,
                .Sampler = ToSamplerConfig(src_sampler),
                .Name = std::format("{} ({})", src_texture.Name.empty() ? std::format("Texture{}", texture_index) : src_texture.Name, color_space == TextureColorSpace::Srgb ? "sRGB" : "Linear"),
            },
        };
        // The manifest records each upload so a snapshot restore can reuse its bindless slot.
        out.Textures.emplace_back(MaterializedTexture{.SamplerSlot = sampler_slot, .SourceImageIndex = *image_index, .Params = upload.Params});
        textures.PendingUploads.emplace_back(upload);
        texture_slot_cache.emplace(cache_key, sampler_slot);
        return sampler_slot;
    };

    out.IndexByGltfMaterial.reserve(source_materials.size());
    std::vector<std::string> material_names;
    material_names.reserve(source_materials.size());
    buffers.Materials.Reserve((buffers.Materials.Count<PBRMaterial>() + source_materials.size()) * sizeof(PBRMaterial));
    for (uint32_t material_index = 0; material_index < source_materials.size(); ++material_index) {
        const auto src_name = material_index < asset.materials.size() ? std::string_view(asset.materials[material_index].name) : std::string_view{"DefaultMaterial"};
        const auto material_name = src_name.empty() ? std::format("Material{}", material_index) : std::string{src_name};
        auto gpu_material = source_materials[material_index];
        for (const auto &slot : MaterialTextureSlots) {
            auto &tex = slot.Get(gpu_material);
            if (tex.Slot == InvalidSlot) continue;
            if (tex.TexCoord > 3u) {
                std::cerr << std::format("Warning: glTF material '{}' texture '{}' uses TEXCOORD_{}. MeshEditor currently supports TEXCOORD_0..3. Clamping to TEXCOORD_3.\n", material_name, slot.Label, tex.TexCoord);
                tex.TexCoord = 3u;
            }
            tex.Slot = resolve_texture_slot(tex.Slot, slot.ColorSpace);
        }
        out.IndexByGltfMaterial.emplace_back(buffers.Materials.Append(gpu_material));
        material_names.emplace_back(material_name);
    }
    r.ctx().get<MaterialStore>().AppendNames(std::move(material_names));
    return out;
}

// Shading features the mesh's materials enable, read from the appended GPU materials.
PbrFeatureMask PbrFeaturesOf(std::span<const PBRMaterial> materials, std::span<const uint32_t> material_indices) {
    PbrFeatureMask mask{0};
    for (const auto index : material_indices) {
        if (index >= materials.size()) continue;
        const auto &mat = materials[index];
        if (mat.Transmission.Factor > 0.f || mat.Transmission.Texture.Slot != InvalidSlot) mask |= PbrFeature::Transmission;
        if (mat.DiffuseTransmission.Factor > 0.f || mat.DiffuseTransmission.Texture.Slot != InvalidSlot) mask |= PbrFeature::DiffuseTrans;
        if (mat.Clearcoat.Factor > 0.f || mat.Clearcoat.Texture.Slot != InvalidSlot) mask |= PbrFeature::Clearcoat;
        if (mat.Sheen.RoughnessFactor > 0.f || mat.Sheen.ColorTexture.Slot != InvalidSlot) mask |= PbrFeature::Sheen;
        if (mat.Anisotropy.Strength != 0.f || mat.Anisotropy.Texture.Slot != InvalidSlot) mask |= PbrFeature::Anisotropy;
        if (mat.Iridescence.Factor > 0.f || mat.Iridescence.Texture.Slot != InvalidSlot) mask |= PbrFeature::Iridescence;
    }
    return mask;
}

// Triangle, line, and point mesh entities per source mesh, indexed by MeshKind.
using MeshEntities = std::vector<std::array<state::Entity, 3>>;

// Remaps primitive materials to their GPU indices, creates every part in source order, and attaches each part's layout.
MeshEntities ImportMeshes(state::Scene &r, std::span<SourceMesh> source_meshes, std::span<const uint32_t> material_index_by_gltf_material, const std::filesystem::path &source_path) {
    const auto remap = [&](uint32_t i) { return i < material_index_by_gltf_material.size() ? material_index_by_gltf_material[i] : material_index_by_gltf_material.back(); };
    struct Part {
        uint32_t Mesh;
        MeshKind Kind;
    };
    std::vector<Part> parts;
    std::vector<MeshSource> sources;
    std::vector<MeshSourceLayout> layouts;
    for (uint32_t mi = 0; mi < source_meshes.size(); ++mi) {
        auto &source_mesh = source_meshes[mi];
        auto &layout = source_mesh.Layout;
        for (auto &material : layout.DefaultMaterials) material = remap(material);
        // KHR_materials_variants mappings then index the same buffer, so applying a variant writes entries straight into it.
        for (auto &mappings : layout.VariantMappings) {
            for (auto &m : mappings) {
                if (m) *m = remap(*m);
            }
        }
        const auto add_part = [&](std::optional<MeshSource> &source, MeshKind kind) {
            if (!source) return;
            source->Primitives.MaterialIndices = layout.DefaultMaterials;
            auto part_layout = layout;
            part_layout.Colors0ComponentCount = source->Attrs.Colors0ComponentCount;
            part_layout.Kind = kind;
            layouts.emplace_back(std::move(part_layout));
            parts.emplace_back(mi, kind);
            sources.emplace_back(std::move(*source));
        };
        add_part(source_mesh.Triangles, MeshKind::Triangles);
        add_part(source_mesh.Lines, MeshKind::Lines);
        add_part(source_mesh.Points, MeshKind::Points);
    }

    auto created = CreateMeshes(r, sources);
    const auto materials = r.ctx().get<const GpuBuffers>().Materials.GetSpan<PBRMaterial>();
    MeshEntities entities(source_meshes.size(), {state::Null, state::Null, state::Null});
    for (uint32_t part = 0; part < parts.size(); ++part) {
        auto &layout = layouts[part];
        // Welding compacts the tangent deltas, so they come from the created mesh.
        layout.MorphTangentDeltas = std::move(created[part].MorphTangentDeltas);
        const auto features = parts[part].Kind == MeshKind::Triangles ? PbrFeaturesOf(materials, layout.DefaultMaterials) : PbrFeatureMask{0};
        const auto [e, _] = ::AddMesh(r, created[part].StoreId, std::nullopt);
        if (!created[part].AuthoredCornerNormals.empty()) r.emplace<AuthoredCornerNormals>(e, std::move(created[part].AuthoredCornerNormals));
        r.emplace<Path>(e, source_path);
        r.emplace<MeshSourceLayout>(e, std::move(layout));
        if (features != 0) r.emplace<PbrMeshFeatures>(e, features);
        entities[parts[part].Mesh][size_t(parts[part].Kind)] = e;
    }
    return entities;
}

struct ImportedObjects {
    // Object entities per node, one per EXT_mesh_gpu_instancing instance and otherwise one.
    std::vector<std::vector<state::Entity>> ByNode;
    state::Entity FirstCamera{state::Null};
    // The entity other nodes reference for a node, the last emitted instance.
    state::Entity Of(uint32_t node_index) const { return node_index < ByNode.size() && !ByNode[node_index].empty() ? ByNode[node_index].back() : state::Null; }
};

// Creates the mesh, camera, light, and empty objects of every emitted node, parents them, and stubs the nodes no scene reaches.
ImportedObjects ImportObjects(state::Scene &r, const fastgltf::Asset &asset, const NodePlan &plan, const MeshEntities &mesh_entities) {
    auto &meshes = r.ctx().get<MeshStore>();
    ImportedObjects objects{.ByNode = std::vector<std::vector<state::Entity>>(asset.nodes.size()), .FirstCamera = state::Null};
    ReserveEntityNames(r, size_t(std::ranges::count(plan.IsObjectEmitted, true)));
    std::vector<bool> instanced(asset.nodes.size(), false);
    for (uint32_t node_index = 0; node_index < asset.nodes.size(); ++node_index) {
        if (!plan.IsObjectEmitted[node_index]) continue;
        const auto &source_node = asset.nodes[node_index];
        const auto mesh_index = ToIndex(source_node.meshIndex, asset.meshes.size());
        const auto camera_index = ToIndex(source_node.cameraIndex, asset.cameras.size());
        const auto light_index = ToIndex(source_node.lightIndex, asset.lights.size());
        // Triangles, then lines, then points instance a mesh whose source lacks triangles.
        const auto primary_mesh = [&]() -> state::Entity {
            if (!mesh_index) return state::Null;
            for (const auto e : mesh_entities[*mesh_index]) {
                if (e != state::Null) return e;
            }
            return state::Null;
        }();
        // EXT_mesh_gpu_instancing emits one root object per instance with its world transform baked in.
        const auto instance_transforms = mesh_index ? ReadInstanceTransforms(asset, source_node) : std::vector<Transform>{};
        instanced[node_index] = !instance_transforms.empty();
        const auto base_name = MakeNodeName(asset, node_index, mesh_index);
        const std::string raw_name{source_node.name};
        const uint32_t count = instanced[node_index] ? uint32_t(instance_transforms.size()) : 1u;
        for (uint32_t i = 0; i < count; ++i) {
            const auto name = instanced[node_index] ? std::format("{}.{}", base_name, i) : base_name;
            const auto transform = instanced[node_index] ? ToTransform(plan.WorldTransforms[node_index] * ToMatrix(instance_transforms[i])) : plan.LocalTransforms[node_index];
            const ObjectCreateInfo info{.Name = name, .Transform = transform, .Select = MeshInstanceCreateInfo::SelectBehavior::None};
            GltfNode node;
            node.Index = node_index;
            state::Entity e = state::Null;
            if (primary_mesh != state::Null) {
                e = ::AddMeshInstance(r, primary_mesh, {.Name = name, .Transform = transform, .Select = MeshInstanceCreateInfo::SelectBehavior::None, .Visible = true});
                // The source mesh's other parts ride under the primary instance with identity transforms.
                for (const auto extra : mesh_entities[*mesh_index]) {
                    if (extra == state::Null || extra == primary_mesh) continue;
                    const auto extra_instance = ::AddMeshInstance(r, extra, {.Name = name, .Transform = Transform{}, .Select = MeshInstanceCreateInfo::SelectBehavior::None, .Visible = true});
                    SetParent(r, extra_instance, e);
                }
            } else if (!mesh_index && camera_index) {
                const auto &cam = asset.cameras[*camera_index];
                e = ::AddCamera(r, meshes, info);
                r.replace<::Camera>(e, ConvertCamera(cam));
                node.Camera = *camera_index;
                node.CameraName = cam.name;
                if (objects.FirstCamera == state::Null) objects.FirstCamera = e;
            } else if (!mesh_index && light_index) {
                const auto &light = asset.lights[*light_index];
                e = ::AddLight(r, meshes, info, ConvertLight(light));
                node.Light = *light_index;
                node.LightName = light.name;
            } else {
                e = ::AddEmpty(r, meshes, info);
            }
            // Record a source name the runtime name replaced or synthesized.
            if (raw_name.empty()) node.EmptyName = true;
            else if (const auto *n = r.try_get<const Name>(e); n && n->Value != raw_name) node.Name = raw_name;
            r.emplace<GltfNode>(e, std::move(node));
            objects.ByNode[node_index].emplace_back(e);
        }
    }

    // Objects nest under their nearest emitted ancestor. Instances are roots, so their baked world transforms stand alone.
    for (uint32_t node_index = 0; node_index < asset.nodes.size(); ++node_index) {
        if (!plan.IsObjectEmitted[node_index] || instanced[node_index]) continue;
        const auto parent = plan.NearestObjectAncestor[node_index];
        if (!parent) continue;
        if (const auto parent_entity = objects.Of(*parent); parent_entity != state::Null) SetParent(r, objects.Of(node_index), parent_entity);
    }

    // Nodes no scene reaches become serialization-only stubs.
    for (uint32_t node_index = 0; node_index < asset.nodes.size(); ++node_index) {
        if (plan.InScene[node_index]) continue;
        const auto &source_node = asset.nodes[node_index];
        const auto e = r.create();
        GltfNode node;
        node.Index = node_index;
        r.emplace<Transform>(e, plan.LocalTransforms[node_index]);
        r.emplace<WorldTransform>(e);
        if (const auto mesh_index = ToIndex(source_node.meshIndex, asset.meshes.size()); mesh_index && mesh_entities[*mesh_index][size_t(MeshKind::Triangles)] != state::Null) {
            r.emplace<Instance>(e, mesh_entities[*mesh_index][size_t(MeshKind::Triangles)]);
        }
        if (source_node.name.empty()) {
            node.EmptyName = true;
        } else {
            const std::string raw_name{source_node.name};
            const auto &name = EmplaceUniqueName(r, e, raw_name);
            if (name.Value != raw_name) node.Name = raw_name;
        }
        r.emplace<GltfNode>(e, std::move(node));
    }
    return objects;
}

PhysicsShape ToPhysicsShape(const fastgltf::Asset &asset, const fastgltf::Geometry &geom) {
    if (geom.shape && *geom.shape < asset.shapes.size()) {
        return std::visit(
            overloaded{
                [](const fastgltf::BoxShape &s) -> PhysicsShape { return physics::Box{ToVec3(s.size)}; },
                [](const fastgltf::SphereShape &s) -> PhysicsShape { return physics::Sphere{s.radius}; },
                [](const fastgltf::CapsuleShape &s) -> PhysicsShape { return physics::Capsule{std::max(float(s.height), physics::MinShapeHeight), s.radiusTop, s.radiusBottom}; },
                [](const fastgltf::CylinderShape &s) -> PhysicsShape { return physics::Cylinder{std::max(float(s.height), physics::MinShapeHeight), s.radiusTop, s.radiusBottom}; },
                [](const fastgltf::PlaneShape &s) -> PhysicsShape { return physics::Plane{s.sizeX, s.sizeZ, s.doubleSided}; },
            },
            asset.shapes[*geom.shape]
        );
    }
    if (geom.convexHull) return physics::ConvexHull{};
    return physics::TriangleMesh{};
}

// Collision filters, then every node's KHR_physics_rigid_bodies collider, motion, trigger, and joint.
void ImportNodePhysics(state::Scene &r, const fastgltf::Asset &asset, const ImportedObjects &objects, const MeshEntities &mesh_entities, const PhysicsResources &resources) {
    // Collision system names dedupe into CollisionSystem entities shared across every filter.
    std::unordered_map<std::string, state::Entity> system_entity_by_name;
    const auto resolve_systems = [&](const auto &names) {
        std::vector<state::Entity> out;
        out.reserve(names.size());
        for (const auto &n : names) {
            std::string key{n};
            auto [it, inserted] = system_entity_by_name.try_emplace(std::move(key), state::Null);
            if (inserted) {
                it->second = r.create();
                r.emplace<CollisionSystem>(it->second, CollisionSystem{.Name = it->first});
            }
            out.emplace_back(it->second);
        }
        return out;
    };
    std::vector<state::Entity> filter_entities;
    filter_entities.reserve(asset.collisionFilters.size());
    for (uint32_t i = 0; i < asset.collisionFilters.size(); ++i) {
        const auto &src = asset.collisionFilters[i];
        // The KHR schema forbids both collideWith and notCollideWith, and the allowlist wins when both appear.
        auto [mode, collide_systems] = [&]() -> std::pair<CollideMode, std::vector<state::Entity>> {
            if (!src.collideWithSystems.empty()) return {CollideMode::Allowlist, resolve_systems(src.collideWithSystems)};
            if (!src.notCollideWithSystems.empty()) return {CollideMode::Blocklist, resolve_systems(src.notCollideWithSystems)};
            return {CollideMode::All, {}};
        }();
        const auto e = r.create();
        r.emplace<CollisionFilter>(e, CollisionFilter{.Systems = resolve_systems(src.collisionSystems), .Mode = mode, .CollideSystems = std::move(collide_systems)});
        r.emplace<SourceIndex>(e, i);
        filter_entities.emplace_back(e);
    }
    const auto resolve_material = [&](const fastgltf::Optional<size_t> &index) {
        const auto i = ToIndex(index, resources.Materials.size());
        return i ? resources.Materials[*i] : state::Null;
    };
    const auto resolve_filter = [&](const fastgltf::Optional<size_t> &index) {
        const auto i = ToIndex(index, filter_entities.size());
        return i ? filter_entities[*i] : state::Null;
    };
    const auto triangles_of = [&](const fastgltf::Optional<size_t> &mesh) {
        const auto i = ToIndex(mesh, asset.meshes.size());
        return i ? mesh_entities[*i][size_t(MeshKind::Triangles)] : state::Null;
    };

    for (uint32_t node_index = 0; node_index < asset.nodes.size(); ++node_index) {
        const auto &rb = asset.nodes[node_index].physicsRigidBody;
        const auto entity = objects.Of(node_index);
        if (!rb || entity == state::Null) continue;

        if (rb->collider) {
            const auto shape = ToPhysicsShape(asset, rb->collider->geometry);
            const auto collider_mesh_entity = [&]() -> state::Entity {
                if (!IsMeshBackedShape(shape)) return state::Null;
                if (const auto geometry_mesh = triangles_of(rb->collider->geometry.mesh); geometry_mesh != state::Null) return geometry_mesh;
                if (const auto *instance = r.try_get<const Instance>(entity)) return instance->Entity;
                return state::Null;
            }();
            r.emplace<ColliderShape>(entity, ColliderShape{.Shape = shape, .MeshEntity = collider_mesh_entity});
            // Imported collider state is authoritative, so the engine never auto-derives over it.
            r.emplace<ColliderPolicy>(entity, ColliderPolicy{.AutoFitDims = false, .LockedKind = true});
            const auto material = resolve_material(rb->collider->physicsMaterial), filter = resolve_filter(rb->collider->collisionFilter);
            if (material != state::Null || filter != state::Null) r.replace<ColliderMaterial>(entity, ColliderMaterial{.PhysicsMaterialEntity = material, .CollisionFilterEntity = filter});
        }
        if (rb->motion) {
            const auto &m = *rb->motion;
            const auto com = ToVec3(m.centerOfMass);
            r.emplace<PhysicsMotion>(entity, PhysicsMotion{
                                                 .IsKinematic = m.isKinematic,
                                                 .Mass = m.mass ? std::optional{float(*m.mass)} : std::nullopt,
                                                 .CenterOfMass = com != vec3{0} ? std::optional{com} : std::nullopt,
                                                 .InertiaDiagonal = m.inertialDiagonal ? std::optional{ToVec3(*m.inertialDiagonal)} : std::nullopt,
                                                 .InertiaOrientation = m.inertialOrientation ? std::optional{std::bit_cast<quat>(*m.inertialOrientation)} : std::nullopt,
                                                 .GravityFactor = float(m.gravityFactor),
                                             });
            if (const auto lv = ToVec3(m.linearVelocity), av = ToVec3(m.angularVelocity); lv != vec3{0} || av != vec3{0}) r.replace<PhysicsVelocity>(entity, PhysicsVelocity{lv, av});
        }
        if (rb->trigger) {
            std::visit(
                overloaded{
                    [&](const fastgltf::GeometryTrigger &t) {
                        // A geometry trigger is a ColliderShape with TriggerTag. KHR makes it exclusive with a solid collider, so a node already carrying one keeps it.
                        if (r.all_of<ColliderShape>(entity)) return;
                        r.emplace<ColliderShape>(entity, ColliderShape{.Shape = ToPhysicsShape(asset, t.geometry), .MeshEntity = triangles_of(t.geometry.mesh)});
                        r.emplace<ColliderPolicy>(entity, ColliderPolicy{.AutoFitDims = false, .LockedKind = true});
                        r.emplace<TriggerTag>(entity);
                        r.patch<ColliderMaterial>(entity, [&](auto &m) { m.CollisionFilterEntity = resolve_filter(t.collisionFilter); });
                    },
                    [&](const fastgltf::NodeTrigger &t) {
                        // A node trigger is a compound zone over other nodes.
                        std::vector<state::Entity> nodes;
                        nodes.reserve(t.nodes.size());
                        for (const auto n : t.nodes) {
                            if (n < asset.nodes.size()) nodes.emplace_back(objects.Of(n));
                        }
                        r.emplace<TriggerNodes>(entity, TriggerNodes{.Nodes = std::move(nodes), .CollisionFilterEntity = state::Null});
                    },
                },
                *rb->trigger
            );
        }
        if (rb->joint) {
            const auto def = ToIndex(rb->joint->joint, resources.JointDefs.size());
            r.emplace<PhysicsJoint>(entity, PhysicsJoint{.ConnectedNode = objects.Of(uint32_t(rb->joint->connectedNode)), .JointDefEntity = def ? resources.JointDefs[*def] : state::Null, .EnableCollision = rb->joint->enableCollision});
        }
    }
}

// KHR_audio_rigid_bodies: modal models, acoustic materials, and acoustic surfaces attached to each instancing node.
void ImportAudio(state::Scene &r, const fastgltf::Asset &asset, const ImportedObjects &objects) {
    if (asset.modalModels.empty() && asset.acousticSurfaces.empty()) return;

    std::vector<AcousticMaterial> acoustic_materials;
    acoustic_materials.reserve(asset.acousticMaterials.size());
    static constexpr auto MaterialDefaults = materials::acoustic::All.front().Properties;
    // Treat out-of-range acoustic values as absent.
    // Reject Poisson ratio 0.5, zero density or modulus, and negative damping to keep derived equations finite and decaying.
    const auto validated = [](const auto &value, double fallback, auto &&ok, std::string_view field, std::string_view name) {
        const double v = value.value_or(fallback);
        if (std::isfinite(v) && ok(v)) return v;
        std::cerr << std::format("Warning: KHR_audio_rigid_bodies acoustic material '{}' has an invalid {} ({}); using {}.\n", name, field, v, fallback);
        return fallback;
    };
    static constexpr auto positive = [](double v) { return v > 0; };
    static constexpr auto non_negative = [](double v) { return v >= 0; };
    for (const auto &m : asset.acousticMaterials) {
        const std::string name{m.name};
        acoustic_materials.emplace_back(AcousticMaterial{
            .Name = name,
            .Properties = {
                .Density = validated(m.density, MaterialDefaults.Density, positive, "density", name),
                .YoungModulus = validated(m.youngsModulus, MaterialDefaults.YoungModulus, positive, "youngsModulus", name),
                .PoissonRatio = validated(m.poissonRatio, MaterialDefaults.PoissonRatio, [](double v) { return v > -1 && v < 0.5; }, "poissonRatio", name),
                .Alpha = validated(m.alpha, MaterialDefaults.Alpha, non_negative, "alpha", name),
                .Beta = validated(m.beta, MaterialDefaults.Beta, non_negative, "beta", name),
            },
        });
    }

    const auto read_accessor = [&]<typename T>(size_t accessor_index) {
        const auto &acc = asset.accessors[accessor_index];
        std::vector<T> out(acc.count);
        fastgltf::copyFromAccessor<T>(asset, acc, out.data());
        return out;
    };
    const auto read_scalars = [&](size_t i) { return read_accessor.template operator()<float>(i); };
    const auto read_vec3s = [&](size_t i) { return read_accessor.template operator()<vec3>(i); };
    const auto read_indices = [&](size_t i) { return read_accessor.template operator()<uint32_t>(i); };

    // Every accessor value has to be finite, since a resonator's state never recovers from a non-finite frequency, decay, or shape.
    const auto all_finite = [](const auto &values) {
        return std::ranges::all_of(values, [](const auto &v) {
            if constexpr (std::is_same_v<std::decay_t<decltype(v)>, vec3>) return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
            else return std::isfinite(v);
        });
    };
    // A model needs its four accessors, one decay rate per mode, and a mode-major M*P shape block, and reads back empty otherwise.
    const auto read_model = [&](const fastgltf::ModalModel &m) -> ModalModes {
        const auto accessors = asset.accessors.size();
        const auto freqs = ToIndex(m.frequencies, accessors), decays = ToIndex(m.decayRates, accessors);
        const auto positions = ToIndex(m.positions, accessors), shapes = ToIndex(m.shapes, accessors);
        if (!freqs || !decays || !positions || !shapes) return {};

        ModalModes modes;
        modes.Freqs = read_scalars(*freqs);
        modes.Positions = read_vec3s(*positions);
        const auto decay_rates = read_scalars(*decays);
        const auto shapes_flat = read_vec3s(*shapes);
        const uint32_t n_modes = modes.Freqs.size(), n_points = modes.Positions.size();
        if (n_modes == 0 || n_points == 0 || decay_rates.size() != n_modes || shapes_flat.size() != size_t(n_modes) * n_points) return {};
        if (!all_finite(modes.Freqs) || !all_finite(decay_rates) || !all_finite(modes.Positions) || !all_finite(shapes_flat)) return {};
        // The spec forbids a frequency at or below zero, and a negative decay rate grows without bound.
        // The shape block is indexed by mode, so one malformed mode invalidates the whole model.
        if (std::ranges::any_of(modes.Freqs, [](float f) { return f <= 0; })) return {};
        if (std::ranges::any_of(decay_rates, [](float d) { return d < 0; })) return {};

        modes.T60s.resize(n_modes);
        for (uint32_t k = 0; k < n_modes; ++k) modes.T60s[k] = decay_rates[k] > 0 ? float(Ln1000 / decay_rates[k]) : 0.f;
        // Shapes arrive mode-major (element m*P + i) and are stored position-major as Shapes[point][mode].
        modes.Shapes.assign(n_points, std::vector<vec3>(n_modes));
        for (uint32_t mode = 0; mode < n_modes; ++mode) {
            for (uint32_t i = 0; i < n_points; ++i) modes.Shapes[i][mode] = shapes_flat[mode * n_points + i];
        }
        // The sample surface is optional, and is dropped unless it describes whole triangles over the model's own sample points.
        if (const auto indices = ToIndex(m.indices, accessors)) {
            auto tris = read_indices(*indices);
            if (tris.size() % 3 == 0 && std::ranges::all_of(tris, [n_points](uint32_t i) { return i < n_points; })) modes.Indices = std::move(tris);
            else std::cerr << std::format("Warning: KHR_audio_rigid_bodies modal model '{}' has sample surface indices outside its sample points; ignoring them.\n", std::string{m.name});
        }
        modes.OriginalFundamentalFreq = modes.Freqs.front();
        return modes;
    };
    // An empty entry keeps the array's indices aligned with the document's while attaching nothing.
    std::vector<ModalModes> models;
    models.reserve(asset.modalModels.size());
    for (const auto &m : asset.modalModels) {
        models.emplace_back(read_model(m));
        if (models.back().Freqs.empty()) {
            std::cerr << std::format("Warning: KHR_audio_rigid_bodies modal model '{}' has accessors that do not match, or a frequency at or below zero, or a negative decay rate; ignoring it.\n", std::string{m.name});
        }
    }

    std::vector<ContactSurface> surfaces;
    surfaces.reserve(asset.acousticSurfaces.size());
    for (const auto &s : asset.acousticSurfaces) {
        static constexpr ContactSurface Defaults{};
        ContactSurface surface{
            .Name = std::string{s.name},
            .Roughness = float(s.roughness.value_or(Defaults.Roughness)),
            .CorrelationLength = float(s.correlationLength.value_or(Defaults.CorrelationLength)),
            .SpectralSlope = float(s.spectralSlope.value_or(Defaults.SpectralSlope)),
            .ShortWavelength = float(s.shortWavelength.value_or(Defaults.ShortWavelength)),
            .Waviness = float(s.waviness.value_or(Defaults.Waviness)),
            .WavinessLength = float(s.wavinessLength.value_or(Defaults.WavinessLength)),
            .Profile = {},
            .SampleSpacing = float(s.sampleSpacing.value_or(0.0)),
            .NormalTexture = {},
        };
        if (const auto profile = ToIndex(s.profile, asset.accessors.size())) surface.Profile = read_scalars(*profile);
        if (const auto texture = s.normalTexture.has_value() ? ToIndex(s.normalTexture->textureIndex, asset.textures.size()) : std::nullopt) {
            surface.NormalTexture = SurfaceNormalTexture{
                .Texture = *texture,
                .TexCoord = uint32_t(s.normalTexture->texCoordIndex),
                .Scale = float(s.normalTexture->scale),
            };
        }
        surfaces.emplace_back(std::move(surface));
    }

    for (uint32_t node_index = 0; node_index < asset.nodes.size(); ++node_index) {
        const auto &source_node = asset.nodes[node_index];
        const auto entity = objects.Of(node_index);
        if (!source_node.audioRigidBody.has_value() || entity == state::Null) continue;
        const auto &instance = *source_node.audioRigidBody;

        const auto surface_index = ToIndex(instance.acousticSurface, asset.acousticSurfaces.size());
        // Store a separate finish on each node that shares a mesh.
        if (surface_index) r.emplace_or_replace<ContactSurface>(entity, surfaces[*surface_index]);

        const auto model_index = [&]() -> std::optional<uint32_t> {
            const auto i = ToIndex(instance.modalModel, models.size());
            return i && !models[*i].Freqs.empty() ? i : std::nullopt;
        }();
        // One acoustic material per node, a model's reference winning over its surface's.
        const auto material_index = [&]() -> std::optional<uint32_t> {
            if (model_index) {
                if (const auto i = ToIndex(asset.modalModels[*model_index].material, acoustic_materials.size())) return i;
            }
            if (surface_index) return ToIndex(asset.acousticSurfaces[*surface_index].material, acoustic_materials.size());
            return std::nullopt;
        }();
        if (material_index) r.emplace_or_replace<AcousticMaterial>(entity, acoustic_materials[*material_index]);

        if (!model_index) continue;
        auto model = models[*model_index];
        model.BakedScale = vec3{1.f};
        // Map each sample point to its nearest render-mesh vertex so the model stays excitable.
        const auto *inst = r.try_get<const Instance>(entity);
        if (inst && r.all_of<MeshHandle>(inst->Entity) && !model.Positions.empty()) {
            const auto mesh = GetMesh(r, inst->Entity);
            model.Vertices.resize(model.Positions.size());
            for (size_t i = 0; i < model.Positions.size(); ++i) {
                uint32_t nearest = 0;
                float nearest_d2 = -1.f;
                for (uint32_t v = 0; v < mesh.VertexCount(); ++v) {
                    const auto d = model.Positions[i] - mesh.GetPosition(Mesh::VH{v});
                    if (const float d2 = numeric::Dot(d, d); nearest_d2 < 0.f || d2 < nearest_d2) {
                        nearest_d2 = d2;
                        nearest = v;
                    }
                }
                model.Vertices[i] = nearest;
            }
        }
        // A model without sample-to-vertex mapping (e.g. a mesh-less node) stays passive data.
        const bool excitable = !model.Vertices.empty();
        r.emplace<ModalModes>(entity, std::move(model));
        if (const auto &mp = asset.modalModels[*model_index].massProperties; mp.has_value()) {
            const auto &q = mp->inertiaOrientation;
            r.emplace<MassProperties>(entity, MassProperties{
                                                  .Mass = mp->mass,
                                                  .CenterOfMass = ToVec3(mp->centerOfMass),
                                                  .InertiaDiagonal = ToVec3(mp->inertiaDiagonal),
                                                  .InertiaOrientation = std::bit_cast<quat>(q),
                                              });
            // Contact dynamics use the dynamic rigid-body mass (UpdateContactDynamics), so a differing modal mass gets a warning.
            // The node's scale sizes the model, so the mass it implies at this size is the solved mass times scale cubed.
            if (const auto *motion = r.try_get<const PhysicsMotion>(entity); motion && IsAuthoritativeDynamicBody(*motion)) {
                const auto *trs = std::get_if<fastgltf::TRS>(&source_node.transform);
                const float node_scale = trs ? MeanScale(std::bit_cast<vec3>(trs->scale)) : 1.f;
                const float sized_mass = float(mp->mass) * node_scale * node_scale * node_scale;
                const float body_mass = motion->Mass.value_or(DefaultMass);
                if (std::abs(body_mass - sized_mass) > 1e-3f * std::max(body_mass, sized_mass)) {
                    const auto name = source_node.name.empty() ? std::format("node {}", node_index) : std::string{source_node.name};
                    std::cerr << std::format(
                        "Warning: '{}': KHR_audio_rigid_bodies mass ({:.4g} kg at the node's scale) disagrees with its KHR_physics_rigid_bodies rigid body ({:.4g} kg); using the rigid body for contact dynamics.\n",
                        name, sized_mass, body_mass
                    );
                }
            }
        } else if (const auto *motion = r.try_get<const PhysicsMotion>(entity); motion && motion->Mass) {
            // A model without its own mass properties falls back to the node's KHR_physics_rigid_bodies motion.
            r.emplace<MassProperties>(entity, MassProperties{
                                                  .Mass = *motion->Mass,
                                                  .CenterOfMass = motion->CenterOfMass.value_or(vec3{0}),
                                                  .InertiaDiagonal = motion->InertiaDiagonal.value_or(vec3{0}),
                                                  .InertiaOrientation = motion->InertiaOrientation.value_or(quat{1, 0, 0, 0}),
                                              });
        }
        if (excitable) r.emplace<SoundVerticesModel>(entity, SoundVerticesModel::Modal);
        if (instance.gain != fastgltf::num(1)) r.emplace<ModalGain>(entity, ModalGain{instance.gain});
    }
}

// Builds each planned armature with its bones, skins, bone instances, and constraints. Returns the armature data entities.
std::vector<state::Entity> ImportArmatures(state::Scene &r, const fastgltf::Asset &asset, const NodePlan &plan, std::span<const ArmaturePlan> groups, const ImportedObjects &objects, std::string_view name_prefix) {
    auto &meshes = r.ctx().get<MeshStore>();
    std::vector<state::Entity> data_entities;
    data_entities.reserve(groups.size());
    for (uint32_t group_index = 0; group_index < groups.size(); ++group_index) {
        const auto &group = groups[group_index];
        const auto arma_node = group.ArmaNode;
        const auto armature_data_entity = r.create();
        auto &armature = r.emplace<Armature>(armature_data_entity);
        data_entities.emplace_back(armature_data_entity);

        std::unordered_map<uint32_t, BoneId> bone_id_by_node;
        bone_id_by_node.reserve(group.BoneNodes.size());
        for (uint32_t k = 0; k < group.BoneNodes.size(); ++k) {
            const auto node = group.BoneNodes[k];
            // Parents precede children, so the parent's bone id is always mapped.
            const auto parent_bone_id = group.BoneParents[k] ? std::optional{bone_id_by_node.at(*group.BoneParents[k])} : std::nullopt;
            const auto bone_id = armature.AddBone(MakeNodeName(asset, node), parent_bone_id, group.RestLocals[k], node);
            bone_id_by_node.emplace(node, bone_id);
            if (const auto object = objects.Of(node);
                object != state::Null && r.all_of<Instance>(object) && !r.all_of<PhysicsMotion>(object) && !r.all_of<BoneAttachment>(object)) {
                r.emplace<BoneAttachment>(object, armature_data_entity, bone_id);
            }
        }

        // Joints stay in source order, so palette slot j pairs with skin.joints[j] and inverseBindMatrices[j], the pairing vertex JOINTS_n attributes index.
        for (const auto skin_index : group.SkinIndices) {
            const auto &skin = asset.skins[skin_index];
            ArmatureImportedSkin imported_skin{
                .SkinIndex = skin_index,
                .SkeletonNodeIndex = ToIndex(skin.skeleton, asset.nodes.size()),
                .AnchorNodeIndex = arma_node,
                .Name = std::string(skin.name),
                .OrderedJointNodeIndices = {},
                .InverseBindMatrices = LoadInverseBindMatrices(asset, skin, skin.joints.size()),
            };
            imported_skin.OrderedJointNodeIndices.reserve(skin.joints.size());
            for (const auto joint_idx : skin.joints) imported_skin.OrderedJointNodeIndices.emplace_back(uint32_t(joint_idx));
            armature.Skins.emplace_back(std::move(imported_skin));
        }
        armature.FinalizeStructure();

        const auto armature_entity = r.create();
        r.emplace<ObjectKind>(armature_entity, ObjectType::Armature);
        r.emplace<ArmatureObject>(armature_entity, armature_data_entity);
        r.emplace<Transform>(armature_entity, arma_node ? ToTransform(plan.WorldTransforms[*arma_node]) : Transform{});
        const auto skin_name = [&]() -> std::string {
            for (const auto skin_index : group.SkinIndices) {
                if (const auto &name = asset.skins[skin_index].name; !name.empty()) return std::string(name);
            }
            return {};
        }();
        EmplaceUniqueName(r, armature_entity, skin_name.empty() ? std::format("{}_Armature{}", name_prefix, group_index) : skin_name);
        GltfNode armature_node;
        armature_node.EmptyName = skin_name.empty();
        r.emplace<GltfNode>(armature_entity, std::move(armature_node));

        // Follow the root node's entity when it is an object (it may be animated), else the nearest object above it.
        if (arma_node) {
            const auto parent_node = objects.Of(*arma_node) != state::Null ? arma_node : plan.NearestObjectAncestor[*arma_node];
            if (const auto parent_entity = parent_node ? objects.Of(*parent_node) : state::Null; parent_entity != state::Null) SetParentKeepWorld(r, armature_entity, parent_entity);
        }

        for (uint32_t skin_slot = 0; skin_slot < group.SkinIndices.size(); ++skin_slot) {
            const auto skin_index = group.SkinIndices[skin_slot];
            // glTF node.skin is deform linkage, not a transform-parent relationship.
            for (uint32_t node_index = 0; node_index < asset.nodes.size(); ++node_index) {
                if (ToIndex(asset.nodes[node_index].skinIndex, asset.skins.size()) != skin_index) continue;
                for (const auto mesh_instance_entity : objects.ByNode[node_index]) {
                    if (!r.all_of<Instance>(mesh_instance_entity)) continue;
                    r.emplace_or_replace<ArmatureModifier>(mesh_instance_entity, armature_data_entity, armature_entity, skin_slot);
                    // The spec ignores a skinned mesh node's own transform.
                    // Identity-parent it to the armature so its world transform is the deform's space.
                    r.emplace_or_replace<Transform>(mesh_instance_entity, Transform{});
                    SetParent(r, mesh_instance_entity, armature_entity);
                }
            }
        }

        // Bone instances only, their pose state is built later from the bone Transforms and rest pose.
        ::CreateBoneInstances(r, meshes, armature_entity, armature_data_entity);
        // Mark each bone entity with its source joint NodeIndex (for SaveScene round-trip).
        const auto &bone_entities = r.get<const ArmatureObject>(armature_entity).BoneEntities;
        for (uint32_t i = 0; i < armature.Bones.size(); ++i) {
            const auto joint_node_index = armature.Bones[i].JointNodeIndex;
            if (!joint_node_index) continue;
            GltfNode bone_node;
            bone_node.Index = *joint_node_index;
            bone_node.EmptyName = *joint_node_index < asset.nodes.size() && asset.nodes[*joint_node_index].name.empty();
            r.emplace<GltfNode>(bone_entities[i], std::move(bone_node));
        }

        // Bones under a physics-driven ancestor get a Child Of constraint so skinned geometry follows simulation.
        // The target is the nearest ancestor object with PhysicsMotion, and InverseMatrix bakes the rest offset.
        const auto find_physics_ancestor_entity = [&](uint32_t node_index) -> state::Entity {
            for (std::optional<uint32_t> cur = node_index; cur; cur = plan.Parents[*cur]) {
                if (const auto object = objects.Of(*cur); object != state::Null && r.all_of<PhysicsMotion>(object)) return object;
            }
            return state::Null;
        };
        const mat4 armature_world = ToMatrix(r.get<const WorldTransform>(armature_entity));
        for (uint32_t i = 0; i < armature.Bones.size(); ++i) {
            const auto &bone = armature.Bones[i];
            if (!bone.JointNodeIndex) continue;
            const auto target = find_physics_ancestor_entity(*bone.JointNodeIndex);
            if (target == state::Null) continue;
            EnsureWorldTransform(r, target);
            r.emplace<BoneConstraints>(bone_entities[i], BoneConstraints{.Stack = {BoneConstraint{
                                                                             .TargetEntity = target,
                                                                             .Influence = 1.f,
                                                                             .Data = ChildOfData{.InverseMatrix = numeric::Inverse(ToMatrix(r.get<const WorldTransform>(target))) * (armature_world * bone.RestWorld)},
                                                                         }}});
        }
    }
    return data_entities;
}

// Records each source-derived entity's source parent, sibling position, and matrix form, then applies KHR_node_visibility.
void RecordSourceHierarchy(state::Scene &r, const fastgltf::Asset &asset, const NodePlan &plan) {
    for (const auto [entity, node] : r.view<const GltfNode>().each()) {
        if (!node.Index || *node.Index >= asset.nodes.size()) continue;
        const auto parent_idx = plan.Parents[*node.Index];
        if (!parent_idx && !plan.SourceMatrices[*node.Index]) continue;
        auto &edited = r.edit<GltfNode>(entity);
        if (parent_idx) {
            edited.Parent = *parent_idx;
            // Sibling position in the parent's bounds-filtered children list.
            uint32_t sibling_idx = 0;
            for (const auto child_raw : asset.nodes[*parent_idx].children) {
                const auto child = ToIndex(child_raw, asset.nodes.size());
                if (!child) continue;
                if (*child == *node.Index) {
                    edited.Sibling = sibling_idx;
                    break;
                }
                ++sibling_idx;
            }
        }
        if (plan.SourceMatrices[*node.Index]) edited.Matrix = *plan.SourceMatrices[*node.Index];
    }

    // KHR_node_visibility: visible:false hides the node and its descendants.
    const auto hide_subtree = [&](this const auto &self, state::Entity e) -> void {
        Hide(r, e);
        for (const auto child : Children{&r, e}) self(child);
    };
    for (const auto [entity, node] : r.view<const GltfNode>().each()) {
        if (node.Index && *node.Index < asset.nodes.size() && !asset.nodes[*node.Index].visible) hide_subtree(entity);
    }
}

struct ImportedAnimations {
    // Source names of animations with at least one valid channel, in source order.
    std::vector<std::string> Order;
    bool Any{false};
};

// Sets up morph weight state, then parses every channel straight into armature, morph, and node clips.
ImportedAnimations ImportAnimations(state::Scene &r, const fastgltf::Asset &asset, state::Entity viewport, const ImportedObjects &objects, std::span<const state::Entity> armature_data_entities) {
    std::unordered_map<uint32_t, std::vector<std::pair<state::Entity, BoneId>>> armature_targets_by_joint_node;
    for (const auto armature_data_entity : armature_data_entities) {
        for (const auto &bone : r.get<const Armature>(armature_data_entity).Bones) {
            if (bone.JointNodeIndex) armature_targets_by_joint_node[*bone.JointNodeIndex].emplace_back(armature_data_entity, bone.Id);
        }
    }

    // Mesh instances with morph targets start at the node's weights, else the mesh defaults. The GPU range (MorphWeightGpuRange) is allocated later.
    const auto &meshes = r.ctx().get<const MeshStore>();
    std::unordered_map<uint32_t, state::Entity> morph_instance_by_node;
    for (uint32_t node_index = 0; node_index < asset.nodes.size(); ++node_index) {
        for (const auto instance_entity : objects.ByNode[node_index]) {
            const auto *instance = r.try_get<const Instance>(instance_entity);
            const auto *handle = instance ? r.try_get<const MeshHandle>(instance->Entity) : nullptr;
            if (!handle) continue;
            const auto &record = meshes.Get(handle->StoreId);
            if (record.MorphTargetCount == 0) continue;
            const auto &node_weights = asset.nodes[node_index].weights;
            auto weights = record.DefaultMorphWeights;
            if (!node_weights.empty()) {
                weights.assign(record.MorphTargetCount, 0.f);
                std::copy_n(node_weights.begin(), std::min(node_weights.size(), size_t(record.MorphTargetCount)), weights.begin());
            }
            r.emplace<MorphWeightState>(instance_entity, MorphWeightState{.Weights = std::move(weights)});
            morph_instance_by_node[node_index] = instance_entity;
        }
    }

    ImportedAnimations out;
    out.Order.reserve(asset.animations.size());
    for (const auto &anim : asset.animations) {
        std::unordered_map<state::Entity, ::AnimationClip> armature_clips_by_entity;
        std::unordered_map<state::Entity, MorphWeightClip> morph_clips_by_entity;
        std::unordered_map<state::Entity, ::AnimationClip> node_clips_by_entity;
        const std::string anim_name(anim.name);
        float max_time = 0;
        bool any_channel = false;

        for (const auto &channel : anim.channels) {
            if (!channel.nodeIndex || *channel.nodeIndex >= asset.nodes.size()) continue;
            if (channel.samplerIndex >= anim.samplers.size()) continue;
            const auto target_node_index = uint32_t(*channel.nodeIndex);
            const auto path = ToPath(channel.path);
            // A weights channel's component count is the target mesh's morph target count.
            const auto component_count = [&]() -> size_t {
                if (path != AnimationPath::Weights) return path == AnimationPath::Rotation ? 4 : 3;
                const auto mesh_index = ToIndex(asset.nodes[target_node_index].meshIndex, asset.meshes.size());
                if (!mesh_index || asset.meshes[*mesh_index].primitives.empty()) return 0;
                return asset.meshes[*mesh_index].primitives[0].targets.size();
            }();
            if (component_count == 0) continue;

            const auto &sampler = anim.samplers[channel.samplerIndex];
            if (sampler.inputAccessor >= asset.accessors.size() || sampler.outputAccessor >= asset.accessors.size()) continue;
            const auto &input_accessor = asset.accessors[sampler.inputAccessor];
            const auto &output_accessor = asset.accessors[sampler.outputAccessor];
            if (input_accessor.count == 0) continue;

            const auto interp = ToInterp(sampler.interpolation);
            std::vector<float> times(input_accessor.count);
            fastgltf::copyFromAccessor<float>(asset, input_accessor, times.data());
            std::vector<float> values;
            if (path == AnimationPath::Weights) {
                values.resize(output_accessor.count);
                fastgltf::copyFromAccessor<float>(asset, output_accessor, values.data());
            } else {
                values.resize(output_accessor.count * component_count);
                if (component_count == 4) fastgltf::copyFromAccessor<vec4>(asset, output_accessor, reinterpret_cast<vec4 *>(values.data()));
                else fastgltf::copyFromAccessor<vec3>(asset, output_accessor, reinterpret_cast<vec3 *>(values.data()));
            }
            max_time = std::max(max_time, times.back());
            any_channel = true;

            if (path == AnimationPath::Weights) {
                const auto inst_it = morph_instance_by_node.find(target_node_index);
                if (inst_it == morph_instance_by_node.end()) continue;
                auto &resolved_clip = morph_clips_by_entity.try_emplace(inst_it->second, MorphWeightClip{.Name = anim_name, .DurationSeconds = 0.f, .Channels = {}}).first->second;
                resolved_clip.Channels.emplace_back(MorphWeightChannel{.Interp = interp, .TimesSeconds = std::move(times), .Values = std::move(values)});
                continue;
            }
            // Channels targeting a joint drive its bone in every armature that owns it.
            if (const auto armature_it = armature_targets_by_joint_node.find(target_node_index); armature_it != armature_targets_by_joint_node.end()) {
                for (const auto &[target_data_entity, bone_id] : armature_it->second) {
                    const auto &armature = r.get<const Armature>(target_data_entity);
                    const auto bone_index = armature.FindBoneIndex(bone_id).value_or(InvalidBoneIndex);
                    auto &resolved_clip = armature_clips_by_entity.try_emplace(target_data_entity, ::AnimationClip{.Name = anim_name, .DurationSeconds = 0.f, .Channels = {}}).first->second;
                    resolved_clip.Channels.emplace_back(::AnimationChannel{.BoneIndex = bone_index, .TargetBoneId = bone_id, .Target = path, .Interp = interp, .TimesSeconds = times, .Values = values});
                }
                continue;
            }
            if (const auto object = objects.Of(target_node_index); object != state::Null) {
                auto &resolved_clip = node_clips_by_entity.try_emplace(object, ::AnimationClip{.Name = anim_name, .DurationSeconds = 0.f, .Channels = {}}).first->second;
                resolved_clip.Channels.emplace_back(::AnimationChannel{.BoneIndex = 0, .Target = path, .Interp = interp, .TimesSeconds = std::move(times), .Values = std::move(values)});
            }
        }

        if (!any_channel) continue;
        out.Order.emplace_back(std::move(anim_name));

        for (auto &[_, c] : armature_clips_by_entity) c.DurationSeconds = max_time;
        for (auto &[_, c] : morph_clips_by_entity) c.DurationSeconds = max_time;
        for (auto &[_, c] : node_clips_by_entity) c.DurationSeconds = max_time;

        for (auto &[target_data_entity, resolved_clip] : armature_clips_by_entity) out.Any |= AppendClip<ArmatureAnimation>(r, target_data_entity, std::move(resolved_clip));
        for (auto &[instance_entity, resolved_clip] : morph_clips_by_entity) out.Any |= AppendClip<MorphWeightAnimation>(r, instance_entity, std::move(resolved_clip));
        for (auto &[object_entity, resolved_clip] : node_clips_by_entity) {
            out.Any = true;
            if (auto *existing = r.try_edit<NodeTransformAnimation>(object_entity)) existing->Clips.emplace_back(std::move(resolved_clip));
            else r.emplace<NodeTransformAnimation>(object_entity, NodeTransformAnimation{.Clips = {std::move(resolved_clip)}, .ActiveClipIndex = 0});
        }
    }

    // The timeline spans the longest imported clip.
    float max_dur = 0;
    for (const auto [_, anim] : r.view<const ArmatureAnimation>().each()) {
        for (const auto &clip : anim.Clips) max_dur = std::max(max_dur, clip.DurationSeconds);
    }
    for (const auto [_, anim] : r.view<const MorphWeightAnimation>().each()) {
        for (const auto &clip : anim.Clips) max_dur = std::max(max_dur, clip.DurationSeconds);
    }
    for (const auto [_, anim] : r.view<const NodeTransformAnimation>().each()) {
        for (const auto &clip : anim.Clips) max_dur = std::max(max_dur, clip.DurationSeconds);
    }
    if (max_dur > 0) r.patch<TimelineRange>(viewport, [&](auto &range) { range.EndFrame = int(std::ceil(max_dur * range.Fps)); });
    return out;
}

// One scene entity per source scene with the default scene active. Multi-scene assets record each node's membership.
void ImportScenes(state::Scene &r, const fastgltf::Asset &asset, uint32_t scene_index, std::span<const uint32_t> scene_mask) {
    std::vector<state::Entity> scene_entities;
    scene_entities.reserve(asset.scenes.size());
    for (uint32_t i = 0; i < asset.scenes.size(); ++i) {
        const auto se = r.create();
        r.emplace<Scene>(se, std::string{asset.scenes[i].name});
        r.emplace<SourceIndex>(se, i);
        if (i == scene_index) r.emplace<ActiveScene>(se);
        scene_entities.emplace_back(se);
    }
    if (asset.scenes.size() <= 1) return;
    for (const auto [e, node] : r.view<const GltfNode>().each()) {
        if (!node.Index || *node.Index >= scene_mask.size()) continue;
        const auto mask = scene_mask[*node.Index];
        std::vector<state::Entity> scenes;
        for (uint32_t i = 0; i < scene_entities.size(); ++i) {
            if (mask & (1u << i)) scenes.emplace_back(scene_entities[i]);
        }
        if (!scenes.empty()) r.emplace<SceneMembership>(e, std::move(scenes));
    }
}
} // namespace

std::expected<fastgltf::Asset, std::string> ParseGltfAsset(const std::filesystem::path &path) { return ParseAsset(path); }

std::expected<LoadResult, std::string> LoadGltf(const std::filesystem::path &source_path, state::Scene &r, state::Entity viewport) {
    const profile::CpuScope scope{"LoadGltf"};

    // Parse and validate everything that can fail before the first entity or store write.
    ExtrasMap extras;
    const auto stored_path = project::ResolveAsset(r, source_path);
    auto parsed_asset = ParseAsset(stored_path, &extras);
    if (!parsed_asset) return std::unexpected{parsed_asset.error()};
    const auto &asset = *parsed_asset;
    if (asset.scenes.empty()) return std::unexpected{std::format("glTF '{}' has no scenes.", source_path.string())};
    const auto scene_index = uint32_t(asset.defaultScene.value_or(0));
    if (scene_index >= asset.scenes.size()) return std::unexpected{std::format("glTF '{}' has invalid default scene index.", source_path.string())};

    auto source_assets = ReadSourceAssets(r, asset, stored_path, std::move(extras), scene_index);
    if (!source_assets) return std::unexpected{std::move(source_assets.error())};
    auto source_materials = ReadMaterials(asset);
    source_assets->MaterialMetas = std::move(source_materials.Metas);

    std::vector<SourceMesh> source_meshes;
    source_meshes.reserve(asset.meshes.size());
    for (uint32_t mesh_index = 0; mesh_index < asset.meshes.size(); ++mesh_index) {
        auto source_mesh = ReadSourceMesh(asset, mesh_index);
        if (!source_mesh) return std::unexpected{std::move(source_mesh.error())};
        source_meshes.emplace_back(std::move(*source_mesh));
    }

    const auto plan = PlanNodes(asset, scene_index);
    const bool any_object = std::ranges::any_of(plan.IsObjectEmitted, [](bool emitted) { return emitted; });
    const bool any_usable_skin = std::ranges::any_of(plan.SkinJointNodes, [](const auto &joints) { return !joints.empty(); });
    if (!any_object && !any_usable_skin) return std::unexpected{std::format("glTF '{}' has no importable source objects or skins.", source_path.string())};
    auto armature_plans = PlanArmatures(asset, plan, source_meshes, source_path);
    if (!armature_plans) return std::unexpected{std::move(armature_plans.error())};

    // Commit. Nothing below can fail.
    const auto physics = ImportPhysicsResources(r, asset);
    if (!asset.materialVariants.empty()) {
        ::MaterialVariants mv;
        mv.Names.reserve(asset.materialVariants.size());
        for (const auto &v : asset.materialVariants) mv.Names.emplace_back(v);
        r.emplace_or_replace<::MaterialVariants>(viewport, std::move(mv));
    } else {
        r.remove<::MaterialVariants>(viewport);
    }
    auto materials = ImportMaterials(r, asset, *source_assets, source_materials.Materials);
    const auto mesh_entities = ImportMeshes(r, source_meshes, materials.IndexByGltfMaterial, source_path);
    const auto objects = ImportObjects(r, asset, plan, mesh_entities);
    ImportNodePhysics(r, asset, objects, mesh_entities, physics);
    ImportAudio(r, asset, objects);
    const auto armature_data_entities = ImportArmatures(r, asset, plan, *armature_plans, objects, source_path.stem().string());
    RecordSourceHierarchy(r, asset, plan);
    auto animations = ImportAnimations(r, asset, viewport, objects, armature_data_entities);
    source_assets->AnimationOrder = std::move(animations.Order);

    auto &environments = r.ctx().get<EnvironmentStore>();
    if (const auto &source_ibl = source_assets->ImageBasedLight) {
        const auto [diffuse_slot, specular_slot] = AllocateIblCubeSlots(r.ctx().get<mtl::BindlessSet>());
        environments.PendingImport = PendingEnvironmentImport{*source_ibl, diffuse_slot, specular_slot};
        environments.ClearRequested = false;
    } else {
        environments.ClearRequested = true;
    }
    // Import-time UX default: show an imported world, hide the (empty) default world.
    // Kept out of the reactive world passes so a snapshot restore reproduces the saved WorldOpacity rather than re-forcing this.
    if (r.all_of<RenderedLighting>(viewport)) r.patch<RenderedLighting>(viewport, [&](auto &l) { l.Value.WorldOpacity = source_assets->ImageBasedLight ? 1.f : 0.f; });

    ImportScenes(r, asset, scene_index, plan.SceneMask);
    ApplySceneVisibility(r);
    ApplyActiveSceneSelection(r);
    if (!materials.Textures.empty()) {
        auto &manifest = r.get_or_emplace<MaterializedTextures>(viewport);
        manifest.Items.insert(manifest.Items.end(), std::make_move_iterator(materials.Textures.begin()), std::make_move_iterator(materials.Textures.end()));
    }
    r.emplace_or_replace<SourceAssets>(viewport, std::move(*source_assets));

    return LoadResult{.FirstCameraObject = objects.FirstCamera, .ImportedAnimation = animations.Any};
}

void SwitchActiveScene(state::Scene &r, state::Entity scene) {
    if (!r.all_of<Scene>(scene) || r.all_of<ActiveScene>(scene)) return;
    r.clear<ActiveScene>();
    r.emplace<ActiveScene>(scene);
    ApplySceneVisibility(r);
    ApplyActiveSceneSelection(r);
}

static_assert(ExtrasCameras == uint32_t(fastgltf::Category::Cameras) && ExtrasMeshes == uint32_t(fastgltf::Category::Meshes) && ExtrasNodes == uint32_t(fastgltf::Category::Nodes) && ExtrasLights == uint32_t(fastgltf::Category::Lights));

std::optional<std::string_view> GetExtras(const SourceAssets &sa, uint32_t category, uint32_t source_index) {
    if (const auto it = sa.ExtrasByEntity.find(ExtrasKey(category, source_index)); it != sa.ExtrasByEntity.end()) return std::string_view{it->second};
    return std::nullopt;
}
} // namespace gltf
