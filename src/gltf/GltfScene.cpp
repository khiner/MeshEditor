#include "GltfScene.h"

#include "TransformMath.h"
#include "Variant.h"
#include "numeric/vec2.h"
#include "numeric/vec3.h"
#include "numeric/vec4.h"
#include "vulkan/Slots.h"

#include <fastgltf/base64.hpp>
#include <fastgltf/core.hpp>
#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/types.hpp>
#include <simdjson.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>
#include <fstream>
#include <limits>
#include <map>
#include <numbers>
#include <numeric>
#include <span>
#include <unordered_map>
#include <unordered_set>

namespace gltf {
namespace {
using ExtrasMap = std::unordered_map<uint64_t, std::string>;

uint64_t ExtrasKey(fastgltf::Category cat, std::size_t idx) {
    return (uint64_t(std::uint32_t(cat)) << 32) | uint64_t(idx);
}
void CollectExtras(simdjson::dom::object *extras, std::size_t idx, fastgltf::Category cat, void *userPtr) {
    if (!extras || !userPtr) return;
    static_cast<ExtrasMap *>(userPtr)->emplace(ExtrasKey(cat, idx), simdjson::minify(*extras));
}
std::optional<std::string> EmitExtras(std::size_t idx, fastgltf::Category cat, void *userPtr) {
    if (!userPtr) return std::nullopt;
    const auto &m = *static_cast<const ExtrasMap *>(userPtr);
    if (const auto it = m.find(ExtrasKey(cat, idx)); it != m.end()) return it->second;
    return std::nullopt;
}

std::optional<uint32_t> ToIndex(std::size_t index, std::size_t upper_bound) {
    if (index >= upper_bound) return {};
    return index;
}
std::optional<uint32_t> ToIndex(const fastgltf::Optional<std::size_t> &index, std::size_t upper_bound) {
    if (!index) return {};
    return ToIndex(*index, upper_bound);
}

Filter ToFilter(fastgltf::Filter filter) {
    switch (filter) {
        case fastgltf::Filter::Nearest: return Filter::Nearest;
        case fastgltf::Filter::Linear: return Filter::Linear;
        case fastgltf::Filter::NearestMipMapNearest: return Filter::NearestMipMapNearest;
        case fastgltf::Filter::LinearMipMapNearest: return Filter::LinearMipMapNearest;
        case fastgltf::Filter::NearestMipMapLinear: return Filter::NearestMipMapLinear;
        case fastgltf::Filter::LinearMipMapLinear: return Filter::LinearMipMapLinear;
    }
    return Filter::LinearMipMapLinear;
}

std::optional<Filter> ToFilter(const fastgltf::Optional<fastgltf::Filter> &filter) {
    if (!filter) return {};
    return ToFilter(*filter);
}

Wrap ToWrap(fastgltf::Wrap wrap) {
    switch (wrap) {
        case fastgltf::Wrap::ClampToEdge: return Wrap::ClampToEdge;
        case fastgltf::Wrap::MirroredRepeat: return Wrap::MirroredRepeat;
        case fastgltf::Wrap::Repeat: return Wrap::Repeat;
    }
    return Wrap::Repeat;
}

MimeType ToMimeType(fastgltf::MimeType mime_type) {
    switch (mime_type) {
        case fastgltf::MimeType::None: return MimeType::None;
        case fastgltf::MimeType::JPEG: return MimeType::JPEG;
        case fastgltf::MimeType::PNG: return MimeType::PNG;
        case fastgltf::MimeType::KTX2: return MimeType::KTX2;
        case fastgltf::MimeType::DDS: return MimeType::DDS;
        case fastgltf::MimeType::GltfBuffer: return MimeType::GltfBuffer;
        case fastgltf::MimeType::OctetStream: return MimeType::OctetStream;
        case fastgltf::MimeType::WEBP: return MimeType::WEBP;
    }
    return MimeType::None;
}

MaterialAlphaMode ToAlphaMode(fastgltf::AlphaMode alpha_mode) {
    switch (alpha_mode) {
        case fastgltf::AlphaMode::Opaque: return MaterialAlphaMode::Opaque;
        case fastgltf::AlphaMode::Mask: return MaterialAlphaMode::Mask;
        case fastgltf::AlphaMode::Blend: return MaterialAlphaMode::Blend;
    }
    return MaterialAlphaMode::Opaque;
}

vec2 ToVec2(const fastgltf::math::nvec2 &v) { return {v.x(), v.y()}; }
vec3 ToVec3(const fastgltf::math::nvec3 &v) { return {v.x(), v.y(), v.z()}; }
vec4 ToVec4(const fastgltf::math::nvec4 &v) { return {v.x(), v.y(), v.z(), v.w()}; }
quat ToQuat(const fastgltf::math::fquat &q) { return {q.w(), q.x(), q.y(), q.z()}; }
Transform TrsToTransform(const fastgltf::TRS &trs) { return {.P = ToVec3(trs.translation), .R = glm::normalize(ToQuat(trs.rotation)), .S = ToVec3(trs.scale)}; }

// Slot = glTF texture index (resolved to a bindless slot later in Scene.cpp). Pass `meta` for
// top-level material textures that need texCoord-override round-trip; nested extension textures
// can omit it.
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
            if (meta) meta->SourceTexCoordOverride = *tc_override;
            out.TexCoord = *tc_override;
        }
    }
    return out;
}

std::expected<std::vector<std::byte>, std::string> ReadFileBytes(const std::filesystem::path &path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return std::unexpected{std::format("Failed to open image file '{}'.", path.string())};

    const auto end = file.tellg();
    if (end <= 0) return std::vector<std::byte>{};
    const auto size = static_cast<std::size_t>(end);
    std::vector<std::byte> bytes(size);
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char *>(bytes.data()), std::streamsize(size));
    if (!file) return std::unexpected{std::format("Failed to read image file '{}'.", path.string())};
    return bytes;
}

std::expected<Image, std::string> ReadImage(const fastgltf::Asset &asset, uint32_t image_index, const std::filesystem::path &base_dir) {
    if (image_index >= asset.images.size()) return std::unexpected{std::format("glTF image index {} is out of range.", image_index)};
    const auto &image = asset.images[image_index];

    Image image_result{
        .Bytes = {},
        .MimeType = MimeType::None,
        .Name = std::string(image.name),
    };

    const auto from_span = [&image_result](const auto &data, fastgltf::MimeType mime_type) {
        image_result.Bytes.resize(data.size());
        std::memcpy(image_result.Bytes.data(), data.data(), data.size());
        image_result.MimeType = ToMimeType(mime_type);
        // fastgltf only sets mimeType when source JSON had the field (pre our magic-byte inference below).
        image_result.SourceHadMimeType = mime_type != fastgltf::MimeType::None;
    };

    auto read_result = std::visit(
        fastgltf::visitor{
            [&](const fastgltf::sources::Array &array) -> std::expected<void, std::string> {
                // With LoadExternalImages off, sources::Array only comes from data URI decode.
                from_span(array.bytes, array.mimeType);
                image_result.SourceDataUri = true;
                return {};
            },
            [&](const fastgltf::sources::Vector &vector) -> std::expected<void, std::string> {
                from_span(vector.bytes, vector.mimeType);
                return {};
            },
            [&](const fastgltf::sources::ByteView &view) -> std::expected<void, std::string> {
                from_span(view.bytes, view.mimeType);
                return {};
            },
            [&](const fastgltf::sources::BufferView &buffer_view) -> std::expected<void, std::string> {
                if (buffer_view.bufferViewIndex >= asset.bufferViews.size()) {
                    return std::unexpected{std::format("glTF image {} references invalid bufferView index {}.", image_index, buffer_view.bufferViewIndex)};
                }
                const auto bytes = fastgltf::DefaultBufferDataAdapter{}(asset, buffer_view.bufferViewIndex);
                from_span(bytes, buffer_view.mimeType);
                return {};
            },
            [&](const fastgltf::sources::URI &uri) -> std::expected<void, std::string> {
                if (!uri.uri.isLocalPath()) {
                    return std::unexpected{std::format("glTF image {} URI '{}' is not a local path.", image_index, uri.uri.string())};
                }
                auto image_path = uri.uri.fspath();
                if (image_path.is_relative()) image_path = base_dir / image_path;
                image_path = image_path.lexically_normal();
                auto bytes = ReadFileBytes(image_path);
                if (!bytes) return std::unexpected{std::move(bytes.error())};
                image_result.Bytes = std::move(*bytes);
                image_result.MimeType = ToMimeType(uri.mimeType);
                image_result.SourceHadMimeType = uri.mimeType != fastgltf::MimeType::None;
                image_result.Uri = std::string(uri.uri.string());
                return {};
            },
            [&](const fastgltf::sources::CustomBuffer &) -> std::expected<void, std::string> {
                return std::unexpected{std::format("glTF image {} uses unsupported custom buffer source.", image_index)};
            },
            [&](const fastgltf::sources::Fallback &) -> std::expected<void, std::string> {
                return std::unexpected{std::format("glTF image {} resolved to fallback image source.", image_index)};
            },
            [&](const std::monostate &) -> std::expected<void, std::string> {
                return std::unexpected{std::format("glTF image {} has no data source.", image_index)};
            },
        },
        image.data
    );
    if (!read_result) return std::unexpected{std::move(read_result.error())};
    if (image_result.MimeType == MimeType::None && image_result.Bytes.size() >= 12) {
        // basist::g_ktx2_file_identifier
        static constexpr uint8_t Ktx2Magic[12]{0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A};
        if (std::memcmp(image_result.Bytes.data(), Ktx2Magic, 12) == 0) image_result.MimeType = MimeType::KTX2;
    }
    return image_result;
}

// Appends positions/edges from a non-triangle primitive into `target`, merging with prior
// primitives of the same topology. NORMAL and COLOR_0 are backfilled with zeros where absent
// so the merged vertex range stays channel-aligned.
void AppendNonTrianglePrimitive(const fastgltf::Asset &asset, const fastgltf::Primitive &primitive, ::MeshData &target, ::MeshVertexAttributes &attrs) {
    const auto position_it = primitive.findAttribute("POSITION");
    if (position_it == primitive.attributes.end()) return;

    const auto &position_accessor = asset.accessors[position_it->accessorIndex];
    if (position_accessor.count == 0) return;

    const uint32_t base_vertex = target.Positions.size();
    target.Positions.resize(base_vertex + position_accessor.count);
    fastgltf::copyFromAccessor<vec3>(asset, position_accessor, &target.Positions[base_vertex]);

    if (const auto normal_it = primitive.findAttribute("NORMAL"); normal_it != primitive.attributes.end()) {
        if (!attrs.Normals) {
            attrs.Normals.emplace();
            attrs.Normals->resize(base_vertex, vec3{0.f});
        }
        const auto &normal_accessor = asset.accessors[normal_it->accessorIndex];
        attrs.Normals->resize(base_vertex + position_accessor.count, vec3{0.f});
        fastgltf::copyFromAccessor<vec3>(asset, normal_accessor, &(*attrs.Normals)[base_vertex]);
    } else if (attrs.Normals) {
        attrs.Normals->resize(base_vertex + position_accessor.count, vec3{0.f});
    }

    // CPU stores vec4 regardless of source; track whether any primitive used VEC3.
    if (const auto color_it = primitive.findAttribute("COLOR_0"); color_it != primitive.attributes.end()) {
        if (!attrs.Colors0) {
            attrs.Colors0.emplace();
            attrs.Colors0->resize(base_vertex, vec4{1.f});
        }
        const auto &color_accessor = asset.accessors[color_it->accessorIndex];
        attrs.Colors0->resize(base_vertex + position_accessor.count, vec4{1.f});
        if (color_accessor.type == fastgltf::AccessorType::Vec3) {
            if (attrs.Colors0ComponentCount == 0) attrs.Colors0ComponentCount = 3;
            std::vector<vec3> colors(position_accessor.count);
            fastgltf::copyFromAccessor<vec3>(asset, color_accessor, colors.data());
            for (uint32_t i = 0; i < position_accessor.count; ++i) {
                (*attrs.Colors0)[base_vertex + i] = vec4{colors[i], 1.f};
            }
        } else if (color_accessor.type == fastgltf::AccessorType::Vec4) {
            attrs.Colors0ComponentCount = 4;
            fastgltf::copyFromAccessor<vec4>(asset, color_accessor, &(*attrs.Colors0)[base_vertex]);
        }
    } else if (attrs.Colors0) {
        attrs.Colors0->resize(base_vertex + position_accessor.count, vec4{1.f});
    }

    if (primitive.type == fastgltf::PrimitiveType::Points) return;

    std::vector<uint32_t> indices;
    if (primitive.indicesAccessor) {
        const auto &index_accessor = asset.accessors[*primitive.indicesAccessor];
        indices.resize(index_accessor.count);
        fastgltf::copyFromAccessor<uint32_t>(asset, index_accessor, indices.data());
    } else {
        indices.resize(position_accessor.count);
        std::iota(indices.begin(), indices.end(), 0u);
    }

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

std::expected<void, std::string> AppendPrimitive(
    const fastgltf::Asset &asset,
    const fastgltf::Primitive &primitive,
    ::MeshData &mesh,
    ::MeshVertexAttributes &attrs,
    std::optional<ArmatureDeformData> &deform,
    std::optional<MorphTargetData> &morph,
    uint32_t &attribute_flags
) {
    attribute_flags = 0;
    if (primitive.type != fastgltf::PrimitiveType::Triangles &&
        primitive.type != fastgltf::PrimitiveType::TriangleStrip &&
        primitive.type != fastgltf::PrimitiveType::TriangleFan) return {};

    const auto position_it = primitive.findAttribute("POSITION");
    if (position_it == primitive.attributes.end()) return {};
    const auto joints_it = primitive.findAttribute("JOINTS_0");
    const auto weights_it = primitive.findAttribute("WEIGHTS_0");
    const bool has_joints = joints_it != primitive.attributes.end();
    const bool has_weights = weights_it != primitive.attributes.end();
    if (has_joints != has_weights) return std::unexpected{"glTF primitive has JOINTS_0 without WEIGHTS_0 (or vice versa)."};

    const auto &position_accessor = asset.accessors[position_it->accessorIndex];
    if (position_accessor.count == 0) return {};

    const uint32_t base_vertex = mesh.Positions.size();
    mesh.Positions.resize(base_vertex + position_accessor.count);
    fastgltf::copyFromAccessor<vec3>(asset, position_accessor, &mesh.Positions[base_vertex]);

    const auto normal_it = primitive.findAttribute("NORMAL");
    const bool has_normals = normal_it != primitive.attributes.end();
    if (has_normals) {
        attribute_flags |= MeshAttributeBit_Normal;
        if (!attrs.Normals) {
            attrs.Normals.emplace();
            attrs.Normals->resize(base_vertex, vec3{0.f});
        }
        const auto &normal_accessor = asset.accessors[normal_it->accessorIndex];
        attrs.Normals->resize(base_vertex + position_accessor.count, vec3{0.f});
        fastgltf::copyFromAccessor<vec3>(asset, normal_accessor, &(*attrs.Normals)[base_vertex]);
    } else if (attrs.Normals) {
        attrs.Normals->resize(base_vertex + position_accessor.count, vec3{0.f});
    }

    const auto tangent_it = primitive.findAttribute("TANGENT");
    const bool has_tangents = tangent_it != primitive.attributes.end();
    if (has_tangents) {
        attribute_flags |= MeshAttributeBit_Tangent;
        if (!attrs.Tangents) {
            attrs.Tangents.emplace();
            attrs.Tangents->resize(base_vertex, vec4{0.f, 0.f, 0.f, 1.f});
        }
        const auto &tangent_accessor = asset.accessors[tangent_it->accessorIndex];
        if (tangent_accessor.count != position_accessor.count) {
            return std::unexpected{std::format("glTF primitive TANGENT count ({}) must match POSITION count ({}).", tangent_accessor.count, position_accessor.count)};
        }
        attrs.Tangents->resize(base_vertex + position_accessor.count, vec4{0.f, 0.f, 0.f, 1.f});
        fastgltf::copyFromAccessor<vec4>(asset, tangent_accessor, &(*attrs.Tangents)[base_vertex]);
    } else if (attrs.Tangents) {
        attrs.Tangents->resize(base_vertex + position_accessor.count, vec4{0.f, 0.f, 0.f, 1.f});
    }

    const auto color_it = primitive.findAttribute("COLOR_0");
    const bool has_colors0 = color_it != primitive.attributes.end();
    if (has_colors0) {
        attribute_flags |= MeshAttributeBit_Color0;
        if (!attrs.Colors0) {
            attrs.Colors0.emplace();
            attrs.Colors0->resize(base_vertex, vec4{1.f});
        }
        const auto &color_accessor = asset.accessors[color_it->accessorIndex];
        if (color_accessor.count != position_accessor.count) {
            return std::unexpected{std::format("glTF primitive COLOR_0 count ({}) must match POSITION count ({}).", color_accessor.count, position_accessor.count)};
        }
        attrs.Colors0->resize(base_vertex + position_accessor.count, vec4{1.f});
        if (color_accessor.type == fastgltf::AccessorType::Vec3) {
            if (attrs.Colors0ComponentCount == 0) attrs.Colors0ComponentCount = 3;
            std::vector<vec3> colors(position_accessor.count);
            fastgltf::copyFromAccessor<vec3>(asset, color_accessor, colors.data());
            for (uint32_t i = 0; i < position_accessor.count; ++i) {
                (*attrs.Colors0)[base_vertex + i] = vec4{colors[i], 1.f};
            }
        } else if (color_accessor.type == fastgltf::AccessorType::Vec4) {
            attrs.Colors0ComponentCount = 4;
            fastgltf::copyFromAccessor<vec4>(asset, color_accessor, &(*attrs.Colors0)[base_vertex]);
        } else {
            return std::unexpected{std::format("glTF primitive COLOR_0 accessor type must be VEC3 or VEC4, got type {}.", int(color_accessor.type))};
        }
    } else if (attrs.Colors0) {
        attrs.Colors0->resize(base_vertex + position_accessor.count, vec4{1.f});
    }

    const auto append_uv_set = [&](uint32_t set_index, std::optional<std::vector<vec2>> &uv_set, uint32_t present_bit) -> std::expected<void, std::string> {
        const auto uv_name = std::format("TEXCOORD_{}", set_index);
        const auto uv_it = primitive.findAttribute(uv_name);
        const bool has_uv = uv_it != primitive.attributes.end();
        if (has_uv) {
            attribute_flags |= present_bit;
            if (!uv_set) {
                uv_set.emplace();
                uv_set->resize(base_vertex, vec2{0.f});
            }
            const auto &uv_accessor = asset.accessors[uv_it->accessorIndex];
            if (uv_accessor.count != position_accessor.count) {
                return std::unexpected{std::format("glTF primitive {} count ({}) must match POSITION count ({}).", uv_name, uv_accessor.count, position_accessor.count)};
            }
            uv_set->resize(base_vertex + position_accessor.count, vec2{0.f});
            fastgltf::copyFromAccessor<vec2>(asset, uv_accessor, &(*uv_set)[base_vertex]);
        } else if (uv_set) {
            uv_set->resize(base_vertex + position_accessor.count, vec2{0.f});
        }
        return {};
    };
    if (auto uv_result = append_uv_set(0, attrs.TexCoords0, MeshAttributeBit_TexCoord0); !uv_result) return uv_result;
    if (auto uv_result = append_uv_set(1, attrs.TexCoords1, MeshAttributeBit_TexCoord1); !uv_result) return uv_result;
    if (auto uv_result = append_uv_set(2, attrs.TexCoords2, MeshAttributeBit_TexCoord2); !uv_result) return uv_result;
    if (auto uv_result = append_uv_set(3, attrs.TexCoords3, MeshAttributeBit_TexCoord3); !uv_result) return uv_result;

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
            const auto j_it = primitive.findAttribute(j_name);
            const auto w_it = primitive.findAttribute(w_name);
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
            for (std::size_t set_index = 0; set_index < sets.size(); ++set_index) {
                auto &s = sets[set_index];
                const auto [j_acc, w_acc] = influence_accessors[set_index];
                s.Joints.resize(position_accessor.count);
                fastgltf::copyFromAccessor<uvec4>(asset, *j_acc, s.Joints.data());
                s.Weights.resize(position_accessor.count);
                fastgltf::copyFromAccessor<vec4>(asset, *w_acc, s.Weights.data());
            }

            // Multiple influence sets: merge all, keep top 4 by weight, renormalize.
            // glTF 2.0 §3.7.3.1: implementations MAY support only 4 influences.
            // https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#skinned-mesh-attributes
            const auto total_influences = sets.size() * 4u;
            std::vector<std::pair<uint32_t, float>> all(total_influences);
            const auto top4_end = all.begin() + 4;
            const auto by_weight = [](const auto &a, const auto &b) { return a.second > b.second; };
            for (std::size_t i = 0; i < position_accessor.count; ++i) {
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

    // Morph targets (blend shapes). Deltas are packed primitive-interleaved here, then repacked
    // into per-target-contiguous layout after all primitives have been appended.
    if (!primitive.targets.empty()) {
        const uint32_t target_count = primitive.targets.size();
        const uint32_t prim_vertex_count = position_accessor.count;
        if (!morph) {
            morph.emplace();
            morph->TargetCount = target_count;
            morph->PositionDeltas.resize(std::size_t(target_count) * base_vertex, vec3{0.f});
        }
        if (morph->TargetCount != target_count) return std::unexpected{"glTF primitive morph target count mismatch between primitives of the same mesh."};

        const auto prev_pos_size = morph->PositionDeltas.size();
        morph->PositionDeltas.resize(prev_pos_size + std::size_t(target_count) * prim_vertex_count, vec3{0.f});
        const auto any_target_has = [&](std::string_view name) {
            for (uint32_t t = 0; t < target_count; ++t) {
                if (primitive.findTargetAttribute(t, name) != primitive.targets[t].end()) return true;
            }
            return false;
        };
        const bool prim_has_normal_deltas = any_target_has("NORMAL");
        const bool prim_has_tangent_deltas = any_target_has("TANGENT");
        if (prim_has_normal_deltas && morph->NormalDeltas.empty() && prev_pos_size > 0) {
            morph->NormalDeltas.resize(prev_pos_size, vec3{0.f});
        }
        if (prim_has_normal_deltas || !morph->NormalDeltas.empty()) {
            const auto prev_norm_size = morph->NormalDeltas.size();
            morph->NormalDeltas.resize(prev_norm_size + std::size_t(target_count) * prim_vertex_count, vec3{0.f});
        }
        if (prim_has_tangent_deltas && morph->TangentDeltas.empty() && prev_pos_size > 0) {
            morph->TangentDeltas.resize(prev_pos_size, vec3{0.f});
        }
        if (prim_has_tangent_deltas || !morph->TangentDeltas.empty()) {
            const auto prev_tan_size = morph->TangentDeltas.size();
            morph->TangentDeltas.resize(prev_tan_size + std::size_t(target_count) * prim_vertex_count, vec3{0.f});
        }
        for (uint32_t t = 0; t < target_count; ++t) {
            if (auto pos_it = primitive.findTargetAttribute(t, "POSITION"); pos_it != primitive.targets[t].end()) {
                const auto &target_accessor = asset.accessors[pos_it->accessorIndex];
                if (target_accessor.count == prim_vertex_count) {
                    fastgltf::copyFromAccessor<vec3>(asset, target_accessor, &morph->PositionDeltas[prev_pos_size + std::size_t(t) * prim_vertex_count]);
                }
            }
            if (!morph->NormalDeltas.empty()) {
                if (auto norm_it = primitive.findTargetAttribute(t, "NORMAL"); norm_it != primitive.targets[t].end()) {
                    const auto &norm_accessor = asset.accessors[norm_it->accessorIndex];
                    const auto prev_norm_size = morph->NormalDeltas.size() - std::size_t(target_count) * prim_vertex_count;
                    if (norm_accessor.count == prim_vertex_count) {
                        fastgltf::copyFromAccessor<vec3>(asset, norm_accessor, &morph->NormalDeltas[prev_norm_size + std::size_t(t) * prim_vertex_count]);
                    }
                }
            }
            if (!morph->TangentDeltas.empty()) {
                if (auto tan_it = primitive.findTargetAttribute(t, "TANGENT"); tan_it != primitive.targets[t].end()) {
                    const auto &tan_accessor = asset.accessors[tan_it->accessorIndex];
                    const auto prev_tan_size = morph->TangentDeltas.size() - std::size_t(target_count) * prim_vertex_count;
                    if (tan_accessor.count == prim_vertex_count) {
                        fastgltf::copyFromAccessor<vec3>(asset, tan_accessor, &morph->TangentDeltas[prev_tan_size + std::size_t(t) * prim_vertex_count]);
                    }
                }
            }
        }
    } else if (morph) {
        const uint32_t prim_vertex_count = position_accessor.count;
        morph->PositionDeltas.resize(morph->PositionDeltas.size() + std::size_t(morph->TargetCount) * prim_vertex_count, vec3{0.f});
        if (!morph->NormalDeltas.empty()) {
            morph->NormalDeltas.resize(morph->NormalDeltas.size() + std::size_t(morph->TargetCount) * prim_vertex_count, vec3{0.f});
        }
        if (!morph->TangentDeltas.empty()) {
            morph->TangentDeltas.resize(morph->TangentDeltas.size() + std::size_t(morph->TargetCount) * prim_vertex_count, vec3{0.f});
        }
    }

    std::vector<uint32_t> indices;
    if (primitive.indicesAccessor) {
        const auto &index_accessor = asset.accessors[*primitive.indicesAccessor];
        indices.resize(index_accessor.count);
        fastgltf::copyFromAccessor<uint32_t>(asset, index_accessor, indices.data());
    } else {
        indices.resize(position_accessor.count);
        std::iota(indices.begin(), indices.end(), 0u);
    }
    if (indices.size() < 3) return {};

    if (primitive.type == fastgltf::PrimitiveType::TriangleStrip) {
        for (uint32_t i = 0; i + 2 < indices.size(); ++i) {
            if (i % 2 == 0) {
                mesh.Faces.emplace_back(std::initializer_list<uint32_t>{base_vertex + indices[i], base_vertex + indices[i + 1], base_vertex + indices[i + 2]});
            } else {
                mesh.Faces.emplace_back(std::initializer_list<uint32_t>{base_vertex + indices[i + 1], base_vertex + indices[i], base_vertex + indices[i + 2]});
            }
        }
    } else if (primitive.type == fastgltf::PrimitiveType::TriangleFan) {
        for (uint32_t i = 1; i + 1 < indices.size(); ++i) {
            mesh.Faces.emplace_back(std::initializer_list<uint32_t>{base_vertex + indices[0], base_vertex + indices[i], base_vertex + indices[i + 1]});
        }
    } else {
        for (uint32_t i = 0; i + 2 < indices.size(); i += 3) {
            mesh.Faces.emplace_back(std::initializer_list<uint32_t>{base_vertex + indices[i], base_vertex + indices[i + 1], base_vertex + indices[i + 2]});
        }
    }
    return {};
}

std::expected<fastgltf::Asset, std::string> ParseAsset(const std::filesystem::path &path, ExtrasMap *extras_out = nullptr) {
    auto gltf_file = fastgltf::MappedGltfFile::FromPath(path);
    if (gltf_file.error() != fastgltf::Error::None) return std::unexpected{std::format("Failed to open glTF file '{}': {}", path.string(), fastgltf::getErrorMessage(gltf_file.error()))};

    static constexpr auto EnabledExtensions = fastgltf::Extensions::KHR_mesh_quantization | fastgltf::Extensions::EXT_mesh_gpu_instancing | fastgltf::Extensions::KHR_lights_punctual | fastgltf::Extensions::EXT_lights_image_based | fastgltf::Extensions::KHR_texture_transform | fastgltf::Extensions::KHR_materials_emissive_strength | fastgltf::Extensions::KHR_materials_unlit | fastgltf::Extensions::KHR_texture_basisu | fastgltf::Extensions::KHR_materials_specular | fastgltf::Extensions::KHR_materials_sheen | fastgltf::Extensions::KHR_materials_ior | fastgltf::Extensions::KHR_materials_dispersion | fastgltf::Extensions::KHR_materials_transmission | fastgltf::Extensions::KHR_materials_diffuse_transmission | fastgltf::Extensions::KHR_materials_volume | fastgltf::Extensions::KHR_materials_clearcoat | fastgltf::Extensions::KHR_materials_anisotropy | fastgltf::Extensions::KHR_materials_iridescence | fastgltf::Extensions::KHR_materials_variants | fastgltf::Extensions::KHR_implicit_shapes | fastgltf::Extensions::KHR_physics_rigid_bodies;
    fastgltf::Parser parser{EnabledExtensions};
    if (extras_out) {
        parser.setUserPointer(extras_out);
        parser.setExtrasParseCallback(CollectExtras);
    }
    using fastgltf::Options;
    // LoadExternalImages off: we want external URIs to stay as sources::URI so ReadImage can
    // preserve the URI form for round-trip (the option loads into sources::Array, losing it).
    // GenerateMeshIndices off: we synthesize iota locally and track per-primitive presence
    // so non-indexed primitives round-trip as non-indexed.
    static constexpr auto ParseOptions = Options::AllowDouble | Options::LoadExternalBuffers;
    auto parsed = parser.loadGltf(gltf_file.get(), path.parent_path(), ParseOptions);
    if (parsed.error() != fastgltf::Error::None) {
        if (parsed.error() == fastgltf::Error::MissingExtensions) {
            // Re-parse with all extensions enabled to recover the full `extensionsRequired` list
            // for the diagnostic - fastgltf bails before populating it on the first pass.
            gltf_file.get().reset();
            fastgltf::Parser probe{static_cast<fastgltf::Extensions>(~0U)};
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
                if (!missing.empty()) return std::unexpected{std::format("Failed to parse glTF '{}': Missing required extensions: {}", path.string(), missing)};
            }
        }
        return std::unexpected{std::format("Failed to parse glTF '{}': {}", path.string(), fastgltf::getErrorMessage(parsed.error()))};
    }

    return std::move(parsed.get());
}

// Always emplaces a MeshData so scene.Meshes stays index-aligned with asset.meshes; callers
// check Triangles/Lines/Points presence before referencing the mesh.
std::expected<uint32_t, std::string> EnsureMeshData(const fastgltf::Asset &asset, uint32_t source_mesh_index, Scene &scene, std::unordered_map<uint32_t, uint32_t> &mesh_index_map) {
    if (const auto it = mesh_index_map.find(source_mesh_index); it != mesh_index_map.end()) return it->second;

    const auto &source_mesh = asset.meshes[source_mesh_index];
    ::MeshData mesh;
    ::MeshVertexAttributes mesh_attrs;
    std::optional<ArmatureDeformData> mesh_deform;
    std::optional<MorphTargetData> mesh_morph;
    ::MeshData lines, points; // merged across all line/point primitives
    ::MeshVertexAttributes line_attrs, point_attrs;
    // Non-triangle primitives contribute 0 vertices here (their verts go into lines/points).
    std::vector<uint32_t> vertex_counts(source_mesh.primitives.size(), 0);
    std::vector<uint32_t> attribute_flags(source_mesh.primitives.size(), 0);
    std::vector<uint8_t> has_source_indices(source_mesh.primitives.size(), 0);
    std::vector<std::vector<std::optional<uint32_t>>> variant_mappings(source_mesh.primitives.size());
    std::vector<uint32_t> face_primitive_indices;
    std::vector<uint32_t> primitive_material_indices(source_mesh.primitives.size(), scene.Materials.empty() ? 0u : uint32_t(scene.Materials.size() - 1u));
    for (uint32_t primitive_index = 0; primitive_index < source_mesh.primitives.size(); ++primitive_index) {
        const auto &primitive = source_mesh.primitives[primitive_index];
        if (const auto material_index = ToIndex(primitive.materialIndex, scene.Materials.size())) {
            primitive_material_indices[primitive_index] = *material_index;
        }
        has_source_indices[primitive_index] = primitive.indicesAccessor.has_value() ? 1u : 0u;
        if (!primitive.mappings.empty()) {
            auto &out = variant_mappings[primitive_index];
            out.reserve(primitive.mappings.size());
            for (const auto &m : primitive.mappings) {
                if (m.has_value()) out.emplace_back(ToIndex(*m, scene.Materials.size()));
                else out.emplace_back(std::nullopt);
            }
        }
        if (primitive.type == fastgltf::PrimitiveType::Points) {
            AppendNonTrianglePrimitive(asset, primitive, points, point_attrs);
            continue;
        }
        if (primitive.type == fastgltf::PrimitiveType::Lines || primitive.type == fastgltf::PrimitiveType::LineStrip || primitive.type == fastgltf::PrimitiveType::LineLoop) {
            AppendNonTrianglePrimitive(asset, primitive, lines, line_attrs);
            continue;
        }
        const uint32_t prev_vertex_count = mesh.Positions.size();
        const uint32_t prev_face_count = mesh.Faces.size();
        if (auto append_result = AppendPrimitive(asset, primitive, mesh, mesh_attrs, mesh_deform, mesh_morph, attribute_flags[primitive_index]); !append_result) {
            return std::unexpected{std::move(append_result.error())};
        }
        vertex_counts[primitive_index] = mesh.Positions.size() - prev_vertex_count;
        const auto appended_face_count = mesh.Faces.size() - prev_face_count;
        face_primitive_indices.insert(face_primitive_indices.end(), appended_face_count, primitive_index);
    }
    const bool has_triangles = !mesh.Positions.empty() && !mesh.Faces.empty();
    const bool has_lines = !lines.Positions.empty();
    const bool has_points = !points.Positions.empty();

    // Repack morph deltas from primitive-interleaved to per-target-contiguous.
    const auto triangle_prim_count = std::ranges::count_if(vertex_counts, [](auto c) { return c > 0; });
    if (mesh_morph && mesh_morph->TargetCount > 0 && triangle_prim_count > 1) {
        const uint32_t total_verts = mesh.Positions.size();
        const auto target_count = mesh_morph->TargetCount;
        const auto repack_channel = [&](std::vector<vec3> &channel) {
            if (channel.empty()) return;
            std::vector<vec3> repacked(std::size_t(target_count) * total_verts, vec3{0.f});
            uint32_t src_off = 0, dst_vert_off = 0;
            for (const auto prim_verts : vertex_counts) {
                if (prim_verts == 0) continue;
                for (uint32_t t = 0; t < target_count; ++t) {
                    for (uint32_t v = 0; v < prim_verts; ++v) {
                        repacked[std::size_t(t) * total_verts + dst_vert_off + v] = channel[src_off + std::size_t(t) * prim_verts + v];
                    }
                }
                src_off += target_count * prim_verts;
                dst_vert_off += prim_verts;
            }
            channel = std::move(repacked);
        };
        repack_channel(mesh_morph->PositionDeltas);
        repack_channel(mesh_morph->NormalDeltas);
        repack_channel(mesh_morph->TangentDeltas);
    }

    // Read default morph target weights from mesh
    if (mesh_morph && !source_mesh.weights.empty()) {
        mesh_morph->DefaultWeights.resize(mesh_morph->TargetCount, 0.f);
        const auto copy_count = std::min(source_mesh.weights.size(), std::size_t(mesh_morph->TargetCount));
        std::copy_n(source_mesh.weights.begin(), copy_count, mesh_morph->DefaultWeights.begin());
    } else if (mesh_morph) {
        mesh_morph->DefaultWeights.assign(mesh_morph->TargetCount, 0.f);
    }

    const auto mesh_index = scene.Meshes.size();
    scene.Meshes.emplace_back(MeshData{
        .Triangles = has_triangles ? std::optional{std::move(mesh)} : std::nullopt,
        .Lines = has_lines ? std::optional{std::move(lines)} : std::nullopt,
        .Points = has_points ? std::optional{std::move(points)} : std::nullopt,
        .TriangleAttrs = std::move(mesh_attrs),
        .LineAttrs = std::move(line_attrs),
        .PointAttrs = std::move(point_attrs),
        .TrianglePrimitives = has_triangles ? ::MeshPrimitives{std::move(face_primitive_indices), std::move(primitive_material_indices), std::move(vertex_counts), std::move(attribute_flags), std::move(has_source_indices), std::move(variant_mappings)} : ::MeshPrimitives{},
        .DeformData = std::move(mesh_deform),
        .MorphData = std::move(mesh_morph),
        .Name = std::string(source_mesh.name),
    });
    mesh_index_map.emplace(source_mesh_index, mesh_index);
    return mesh_index;
}

std::vector<std::optional<uint32_t>> BuildNodeParentTable(const fastgltf::Asset &asset) {
    std::vector<std::optional<uint32_t>> parents(asset.nodes.size(), std::nullopt);
    for (std::size_t parent_idx = 0; parent_idx < asset.nodes.size(); ++parent_idx) {
        const uint32_t parent = parent_idx;
        for (const auto child_idx : asset.nodes[parent_idx].children) {
            const auto child = ToIndex(child_idx, asset.nodes.size());
            if (!child || parents[*child]) continue;
            parents[*child] = parent;
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

std::optional<uint32_t> FindNearestJointAncestor(uint32_t node_index, const std::vector<std::optional<uint32_t>> &parents, const std::vector<bool> &is_joint_in_skin) {
    auto parent = parents[node_index];
    while (parent) {
        if (is_joint_in_skin[*parent]) return parent;
        parent = parents[*parent];
    }
    return {};
}

std::optional<uint32_t> FindNearestEmittedObjectAncestor(uint32_t node_index, const std::vector<std::optional<uint32_t>> &parents, const std::vector<bool> &is_object_emitted) {
    auto parent = parents[node_index];
    while (parent) {
        if (is_object_emitted[*parent]) return parent;
        parent = parents[*parent];
    }
    return {};
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
        std::size_t prefix = 0;
        while (prefix < common_count && common_path[prefix] == path[prefix]) ++prefix;
        common_path.resize(prefix);
    }

    if (common_path.empty()) return {};
    return common_path.back();
}

std::expected<Transform, std::string> ComputeJointRestLocal(
    uint32_t skin_index,
    uint32_t joint_node_index,
    std::optional<uint32_t> parent_joint_node_index,
    std::optional<uint32_t> anchor_node_index,
    const std::vector<std::optional<uint32_t>> &parents,
    const std::vector<Transform> &local_transforms
) {
    const auto rebased_parent_node_index = parent_joint_node_index ? parent_joint_node_index : anchor_node_index;
    if (!rebased_parent_node_index) return local_transforms[joint_node_index];
    if (*rebased_parent_node_index == joint_node_index) return Transform{};

    mat4 rebased_local{I4};
    auto current = joint_node_index;
    while (current != *rebased_parent_node_index) {
        rebased_local = ToMatrix(local_transforms[current]) * rebased_local;
        if (!parents[current]) {
            return std::unexpected{std::format("glTF skin {} joint node {} cannot be rebased to ancestor node {}.", skin_index, joint_node_index, *rebased_parent_node_index)};
        }
        current = *parents[current];
    }
    return ToTransform(rebased_local);
}

std::expected<std::vector<uint32_t>, std::string> BuildParentBeforeChildJointOrder(const std::vector<uint32_t> &source_joint_nodes, const std::unordered_map<uint32_t, std::optional<uint32_t>> &joint_parent_map, uint32_t skin_index) {
    std::vector<uint32_t> ordered;
    ordered.reserve(source_joint_nodes.size());

    std::unordered_map<uint32_t, uint8_t> state;
    state.reserve(source_joint_nodes.size());

    const auto emit_joint = [&](uint32_t joint_node_index, const auto &self) -> std::expected<void, std::string> {
        const auto current_state = state[joint_node_index];
        if (current_state == 2) return {};
        if (current_state == 1) return std::unexpected{std::format("glTF skin {} has a cycle in joint ancestry at node {}.", skin_index, joint_node_index)};

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
    if (!node.name.empty()) return std::string(node.name);

    if (source_mesh_index && *source_mesh_index < asset.meshes.size()) {
        const auto &mesh_name = asset.meshes[*source_mesh_index].name;
        if (!mesh_name.empty()) return std::string(mesh_name);
    }

    return std::format("Node{}", node_index);
}
std::vector<Transform> ReadInstanceTransforms(const fastgltf::Asset &asset, const fastgltf::Node &node) {
    if (node.instancingAttributes.empty()) return {};

    const auto t_attr = node.findInstancingAttribute("TRANSLATION");
    const auto r_attr = node.findInstancingAttribute("ROTATION");
    const auto s_attr = node.findInstancingAttribute("SCALE");

    // Determine instance count from the first present accessor
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
            transforms[i].R = glm::normalize(quat{v.w, v.x, v.y, v.z});
        });
    }
    if (s_attr != node.instancingAttributes.end()) {
        const auto &accessor = asset.accessors[s_attr->accessorIndex];
        fastgltf::iterateAccessorWithIndex<vec3>(asset, accessor, [&](const vec3 &v, auto i) { transforms[i].S = v; });
    }

    return transforms;
}
} // namespace

MaterialData FromGpu(const PBRMaterial &m) {
    MaterialData d{
        .BaseColorFactor = m.BaseColorFactor,
        .EmissiveFactor = m.EmissiveFactor,
        .MetallicFactor = m.MetallicFactor,
        .RoughnessFactor = m.RoughnessFactor,
        .NormalScale = m.NormalScale,
        .OcclusionStrength = m.OcclusionStrength,
        .AlphaMode = m.AlphaMode,
        .AlphaCutoff = m.AlphaCutoff,
        .DoubleSided = m.DoubleSided,
        .Unlit = m.Unlit,
        .BaseColorTexture = m.BaseColorTexture,
        .MetallicRoughnessTexture = m.MetallicRoughnessTexture,
        .NormalTexture = m.NormalTexture,
        .OcclusionTexture = m.OcclusionTexture,
        .EmissiveTexture = m.EmissiveTexture,
    };
    // Emit extension blocks only when their values diverge from spec defaults — otherwise the
    // extension block was either absent in source or carried only defaults (filtered by the
    // existing default-omission expected-divergence list).
    if (m.Ior != 1.5f) d.Ior = m.Ior;
    if (m.Dispersion != 0.f) d.Dispersion = m.Dispersion;
    if (m.Sheen != ::Sheen{}) d.Sheen = m.Sheen;
    if (m.Specular != ::Specular{}) d.Specular = m.Specular;
    if (m.Transmission != ::Transmission{}) d.Transmission = m.Transmission;
    if (m.DiffuseTransmission != ::DiffuseTransmission{}) d.DiffuseTransmission = m.DiffuseTransmission;
    if (m.Volume != ::Volume{}) d.Volume = m.Volume;
    if (m.Clearcoat != ::Clearcoat{}) d.Clearcoat = m.Clearcoat;
    if (m.Anisotropy != ::Anisotropy{}) d.Anisotropy = m.Anisotropy;
    if (m.Iridescence != ::Iridescence{}) d.Iridescence = m.Iridescence;
    return d;
}

PBRMaterial ToGpu(const MaterialData &m) {
    const float emissive_strength = m.EmissiveStrength.value_or(1.f);
    return PBRMaterial{
        .BaseColorFactor = m.BaseColorFactor,
        .EmissiveFactor = m.EmissiveFactor * emissive_strength,
        .MetallicFactor = m.MetallicFactor,
        .RoughnessFactor = m.RoughnessFactor,
        .NormalScale = m.NormalScale,
        .OcclusionStrength = m.OcclusionStrength,
        .AlphaMode = m.AlphaMode,
        .AlphaCutoff = m.AlphaCutoff,
        .DoubleSided = m.DoubleSided,
        .Unlit = m.Unlit,
        .Ior = m.Ior.value_or(1.5f),
        .Dispersion = m.Dispersion.value_or(0.f),
        .BaseColorTexture = m.BaseColorTexture,
        .MetallicRoughnessTexture = m.MetallicRoughnessTexture,
        .NormalTexture = m.NormalTexture,
        .OcclusionTexture = m.OcclusionTexture,
        .EmissiveTexture = m.EmissiveTexture,
        .Sheen = m.Sheen.value_or(::Sheen{}),
        .Specular = m.Specular.value_or(::Specular{}),
        .Transmission = m.Transmission.value_or(::Transmission{}),
        .DiffuseTransmission = m.DiffuseTransmission.value_or(::DiffuseTransmission{}),
        .Volume = m.Volume.value_or(::Volume{}),
        .Clearcoat = m.Clearcoat.value_or(::Clearcoat{}),
        .Anisotropy = m.Anisotropy.value_or(::Anisotropy{}),
        .Iridescence = m.Iridescence.value_or(::Iridescence{}),
    };
}

std::expected<Scene, std::string> LoadScene(const std::filesystem::path &path) {
    ExtrasMap extras;
    auto parsed_asset = ParseAsset(path, &extras);
    if (!parsed_asset) return std::unexpected{parsed_asset.error()};

    auto &asset = *parsed_asset;
    if (asset.scenes.empty()) return std::unexpected{std::format("glTF '{}' has no scenes.", path.string())};

    const auto scene_index = asset.defaultScene.value_or(0);
    if (scene_index >= asset.scenes.size()) return std::unexpected{std::format("glTF '{}' has invalid default scene index.", path.string())};

    Scene scene;
    scene.ExtrasByEntity = std::move(extras);
    if (asset.assetInfo) {
        scene.Copyright = std::string(asset.assetInfo->copyright);
        scene.Generator = std::string(asset.assetInfo->generator);
        scene.MinVersion = std::string(asset.assetInfo->minVersion);
        scene.AssetExtras = std::string(asset.assetInfo->extras);
        scene.AssetExtensions = std::string(asset.assetInfo->extensions);
    }
    scene.ExtensionsRequired.reserve(asset.extensionsRequired.size());
    for (const auto &e : asset.extensionsRequired) scene.ExtensionsRequired.emplace_back(e);
    scene.MaterialVariants.reserve(asset.materialVariants.size());
    for (const auto &v : asset.materialVariants) scene.MaterialVariants.emplace_back(v);
    scene.DefaultSceneName = std::string(asset.scenes[scene_index].name);
    scene.DefaultSceneRoots.reserve(asset.scenes[scene_index].nodeIndices.size());
    for (const auto n : asset.scenes[scene_index].nodeIndices) {
        if (const auto idx = ToIndex(n, asset.nodes.size())) scene.DefaultSceneRoots.emplace_back(*idx);
    }
    if (const auto ibl_idx = ToIndex(asset.scenes[scene_index].imageBasedLightIndex, asset.imageBasedLights.size())) {
        const auto &src_ibl = asset.imageBasedLights[*ibl_idx];
        ImageBasedLight ibl{
            .Rotation = glm::normalize(quat{src_ibl.rotation[3], src_ibl.rotation[0], src_ibl.rotation[1], src_ibl.rotation[2]}),
            .SpecularImageSize = src_ibl.specularImageSize,
            .Intensity = std::max(0.f, src_ibl.intensity),
            .Name = src_ibl.name.empty() ? std::format("ImageBasedLight{}", *ibl_idx) : std::string(src_ibl.name),
        };
        ibl.SpecularImageIndicesByMip.reserve(src_ibl.specularImages.size());
        for (const auto &mip : src_ibl.specularImages) {
            std::array<uint32_t, 6> faces{};
            for (std::size_t face = 0; face < 6; ++face) faces[face] = mip[face];
            ibl.SpecularImageIndicesByMip.emplace_back(faces);
        }
        if (src_ibl.irradianceCoefficients) {
            std::array<vec3, 9> coefficients{};
            for (std::size_t i = 0; i < 9; ++i) {
                coefficients[i] = {(*src_ibl.irradianceCoefficients)[i][0], (*src_ibl.irradianceCoefficients)[i][1], (*src_ibl.irradianceCoefficients)[i][2]};
            }
            ibl.IrradianceCoefficients = coefficients;
        }
        scene.ImageBasedLight = std::move(ibl);
    }

    scene.Samplers.reserve(asset.samplers.size());
    for (const auto &sampler : asset.samplers) {
        scene.Samplers.emplace_back(Sampler{
            .MagFilter = ToFilter(sampler.magFilter),
            .MinFilter = ToFilter(sampler.minFilter),
            .WrapS = ToWrap(sampler.wrapS),
            .WrapT = ToWrap(sampler.wrapT),
            .Name = std::string(sampler.name),
        });
    }

    scene.Images.reserve(asset.images.size());
    for (uint32_t image_index = 0; image_index < asset.images.size(); ++image_index) {
        auto image_result = ReadImage(asset, image_index, path.parent_path());
        if (!image_result) return std::unexpected{std::move(image_result.error())};
        scene.Images.emplace_back(std::move(*image_result));
    }

    scene.Textures.reserve(asset.textures.size());
    for (const auto &texture : asset.textures) {
        scene.Textures.emplace_back(Texture{
            .SamplerIndex = ToIndex(texture.samplerIndex, asset.samplers.size()),
            .ImageIndex = ToIndex(texture.imageIndex, asset.images.size()),
            .WebpImageIndex = ToIndex(texture.webpImageIndex, asset.images.size()),
            .BasisuImageIndex = ToIndex(texture.basisuImageIndex, asset.images.size()),
            .DdsImageIndex = ToIndex(texture.ddsImageIndex, asset.images.size()),
            .Name = std::string(texture.name),
        });
    }

    scene.Materials.reserve(asset.materials.size() + 1u);
    for (uint32_t material_index = 0; material_index < asset.materials.size(); ++material_index) {
        const auto &material = asset.materials[material_index];
        TextureTransformMeta base_meta{}, mr_meta{}, normal_meta{}, occlusion_meta{}, emissive_meta{};
        MaterialData data{
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
            .BaseColorTexture = ToTextureIndex(material.pbrData.baseColorTexture, asset, &base_meta),
            .MetallicRoughnessTexture = ToTextureIndex(material.pbrData.metallicRoughnessTexture, asset, &mr_meta),
            .NormalTexture = ToTextureIndex(material.normalTexture, asset, &normal_meta),
            .OcclusionTexture = ToTextureIndex(material.occlusionTexture, asset, &occlusion_meta),
            .EmissiveTexture = ToTextureIndex(material.emissiveTexture, asset, &emissive_meta),
            .BaseColorMeta = base_meta,
            .MetallicRoughnessMeta = mr_meta,
            .NormalMeta = normal_meta,
            .OcclusionMeta = occlusion_meta,
            .EmissiveMeta = emissive_meta,
        };
        if (material.ior) data.Ior = float(*material.ior);
        if (material.dispersion) data.Dispersion = float(*material.dispersion);
        if (material.emissiveStrength) data.EmissiveStrength = float(*material.emissiveStrength);

        if (material.sheen) {
            data.Sheen = ::Sheen{
                .ColorFactor = ToVec3(material.sheen->sheenColorFactor),
                .RoughnessFactor = material.sheen->sheenRoughnessFactor,
                .ColorTexture = ToTextureIndex(material.sheen->sheenColorTexture, asset),
                .RoughnessTexture = ToTextureIndex(material.sheen->sheenRoughnessTexture, asset),
            };
        }
        if (material.specular) {
            data.Specular = ::Specular{
                .Factor = material.specular->specularFactor,
                .ColorFactor = ToVec3(material.specular->specularColorFactor),
                .Texture = ToTextureIndex(material.specular->specularTexture, asset),
                .ColorTexture = ToTextureIndex(material.specular->specularColorTexture, asset),
            };
        }
        if (material.transmission) {
            data.Transmission = ::Transmission{
                .Factor = material.transmission->transmissionFactor,
                .Texture = ToTextureIndex(material.transmission->transmissionTexture, asset),
            };
        }
        if (material.diffuseTransmission) {
            data.DiffuseTransmission = ::DiffuseTransmission{
                .Factor = material.diffuseTransmission->diffuseTransmissionFactor,
                .ColorFactor = ToVec3(material.diffuseTransmission->diffuseTransmissionColorFactor),
                .Texture = ToTextureIndex(material.diffuseTransmission->diffuseTransmissionTexture, asset),
                .ColorTexture = ToTextureIndex(material.diffuseTransmission->diffuseTransmissionColorTexture, asset),
            };
        }
        if (material.volume) {
            const float ad = material.volume->attenuationDistance;
            data.Volume = ::Volume{
                .ThicknessFactor = material.volume->thicknessFactor,
                .AttenuationColor = ToVec3(material.volume->attenuationColor),
                .AttenuationDistance = (std::isinf(ad) || ad <= 0.f) ? 0.f : ad,
                .ThicknessTexture = ToTextureIndex(material.volume->thicknessTexture, asset),
            };
        }
        if (material.clearcoat) {
            data.Clearcoat = ::Clearcoat{
                .Factor = material.clearcoat->clearcoatFactor,
                .RoughnessFactor = material.clearcoat->clearcoatRoughnessFactor,
                .NormalScale = material.clearcoat->clearcoatNormalTexture ? material.clearcoat->clearcoatNormalTexture->scale : 1.f,
                .Texture = ToTextureIndex(material.clearcoat->clearcoatTexture, asset),
                .RoughnessTexture = ToTextureIndex(material.clearcoat->clearcoatRoughnessTexture, asset),
                .NormalTexture = ToTextureIndex(material.clearcoat->clearcoatNormalTexture, asset),
            };
        }
        if (material.anisotropy) {
            data.Anisotropy = ::Anisotropy{
                .Strength = material.anisotropy->anisotropyStrength,
                .Rotation = material.anisotropy->anisotropyRotation,
                .Texture = ToTextureIndex(material.anisotropy->anisotropyTexture, asset),
            };
        }
        if (material.iridescence) {
            data.Iridescence = ::Iridescence{
                .Factor = material.iridescence->iridescenceFactor,
                .Ior = material.iridescence->iridescenceIor,
                .ThicknessMinimum = material.iridescence->iridescenceThicknessMinimum,
                .ThicknessMaximum = material.iridescence->iridescenceThicknessMaximum,
                .Texture = ToTextureIndex(material.iridescence->iridescenceTexture, asset),
                .ThicknessTexture = ToTextureIndex(material.iridescence->iridescenceThicknessTexture, asset),
            };
        }

        scene.Materials.emplace_back(NamedMaterial{
            .Value = std::move(data),
            .Name = std::string(material.name),
        });
    }
    scene.Materials.emplace_back(NamedMaterial{.Name = "DefaultMaterial"});

    const auto parents = BuildNodeParentTable(asset);
    std::vector<Transform> local_transforms(asset.nodes.size());
    std::vector<std::optional<mat4>> source_matrices(asset.nodes.size());
    for (uint32_t node_index = 0; node_index < asset.nodes.size(); ++node_index) {
        const auto &fg_transform = asset.nodes[node_index].transform;
        if (std::holds_alternative<fastgltf::TRS>(fg_transform)) {
            local_transforms[node_index] = TrsToTransform(std::get<fastgltf::TRS>(fg_transform));
        } else {
            const auto &fm = std::get<fastgltf::math::fmat4x4>(fg_transform);
            mat4 m{};
            for (std::size_t c = 0; c < 4; ++c) {
                for (std::size_t r = 0; r < 4; ++r) m[c][r] = fm[c][r];
            }
            source_matrices[node_index] = m;

            fastgltf::math::fvec3 scale, translation;
            fastgltf::math::fquat rotation;
            fastgltf::math::decomposeTransformMatrix(fm, scale, rotation, translation);
            local_transforms[node_index] = Transform{ToVec3(translation), glm::normalize(ToQuat(rotation)), ToVec3(scale)};
        }
    }
    const auto traversal = TraverseSceneNodes(asset, local_transforms, uint32_t(scene_index));

    std::vector<bool> used_skin(asset.skins.size(), false);
    for (uint32_t node_index = 0; node_index < asset.nodes.size(); ++node_index) {
        if (!traversal.InScene[node_index]) continue;
        if (const auto skin_index = ToIndex(asset.nodes[node_index].skinIndex, asset.skins.size())) used_skin[*skin_index] = true;
    }

    // Parse cameras
    scene.Cameras.reserve(asset.cameras.size());
    for (const auto &cam : asset.cameras) {
        auto projection = std::visit(
            [](const auto &source) -> ::Camera {
                using Projection = std::decay_t<decltype(source)>;
                if constexpr (std::is_same_v<Projection, fastgltf::Camera::Perspective>) {
                    return Perspective{.FieldOfViewRad = source.yfov, .FarClip = source.zfar, .NearClip = source.znear, .AspectRatio = source.aspectRatio};
                } else {
                    return Orthographic{.Mag = {source.xmag, source.ymag}, .FarClip = source.zfar, .NearClip = source.znear};
                }
            },
            cam.camera
        );
        scene.Cameras.emplace_back(std::move(projection), std::string{cam.name});
    }

    // Parse KHR_lights_punctual lights
    scene.Lights.reserve(asset.lights.size());
    for (const auto &light : asset.lights) {
        PunctualLight punctual_light{
            .Range = 0.f,
            .Color = ToVec3(light.color),
            .Intensity = light.intensity,
            .InnerConeCos = 0.f,
            .OuterConeCos = 0.f,
            .Type = PunctualLightType::Point,
        };
        switch (light.type) {
            case fastgltf::LightType::Directional:
                punctual_light.Type = PunctualLightType::Directional;
                break;
            case fastgltf::LightType::Point:
                punctual_light.Type = PunctualLightType::Point;
                punctual_light.Range = light.range ? *light.range : 0.f;
                break;
            case fastgltf::LightType::Spot: {
                const float outer = light.outerConeAngle ? *light.outerConeAngle : std::numbers::pi_v<float> / 4.f;
                const float inner = std::clamp(light.innerConeAngle ? *light.innerConeAngle : 0.f, 0.f, outer);
                punctual_light.Type = PunctualLightType::Spot;
                punctual_light.Range = light.range ? *light.range : 0.f;
                punctual_light.InnerConeCos = std::cos(inner);
                punctual_light.OuterConeCos = std::cos(outer);
                break;
            }
        }
        scene.Lights.emplace_back(punctual_light, std::string{light.name});
    }

    // Parse KHR_physics_rigid_bodies document-level resources
    {
        const auto ToCombineMode = [](fastgltf::CombineMode m) {
            switch (m) {
                case fastgltf::CombineMode::Minimum: return PhysicsCombineMode::Minimum;
                case fastgltf::CombineMode::Maximum: return PhysicsCombineMode::Maximum;
                case fastgltf::CombineMode::Multiply: return PhysicsCombineMode::Multiply;
                default: return PhysicsCombineMode::Average;
            }
        };
        for (const auto &src : asset.physicsMaterials) {
            scene.PhysicsMaterials.emplace_back(PhysicsMaterial{
                .StaticFriction = float(src.staticFriction),
                .DynamicFriction = float(src.dynamicFriction),
                .Restitution = float(src.restitution),
                .FrictionCombine = ToCombineMode(src.frictionCombine),
                .RestitutionCombine = ToCombineMode(src.restitutionCombine),
            });
        }
        for (const auto &src : asset.collisionFilters) {
            CollisionFilterData filter;
            for (const auto &s : src.collisionSystems) filter.CollisionSystems.emplace_back(s);
            for (const auto &s : src.collideWithSystems) filter.CollideWithSystems.emplace_back(s);
            for (const auto &s : src.notCollideWithSystems) filter.NotCollideWithSystems.emplace_back(s);
            scene.CollisionFilters.emplace_back(std::move(filter));
        }
        for (const auto &src : asset.physicsJoints) {
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
                // fastgltf zero-initializes maxForce when absent; KHR spec defaults to FLT_MAX.
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
            scene.PhysicsJointDefs.emplace_back(std::move(def));
        }
    }

    // Convert implicit shapes for physics geometry references.
    // Mesh-backed shapes leave MeshEntity as null_entity here - SceneGltf.cpp resolves it via node-to-entity mapping.
    const auto ToPhysicsShape = [&](const fastgltf::Geometry &geom) -> PhysicsShape {
        if (geom.shape && *geom.shape < asset.shapes.size()) {
            return std::visit(
                overloaded{
                    [](const fastgltf::BoxShape &s) -> PhysicsShape { return physics::Box{ToVec3(s.size)}; },
                    [](const fastgltf::SphereShape &s) -> PhysicsShape { return physics::Sphere{float(s.radius)}; },
                    [](const fastgltf::CapsuleShape &s) -> PhysicsShape {
                        return physics::Capsule{float(s.height), float(s.radiusTop), float(s.radiusBottom)};
                    },
                    [](const fastgltf::CylinderShape &s) -> PhysicsShape {
                        return physics::Cylinder{float(s.height), float(s.radiusTop), float(s.radiusBottom)};
                    },
                    [](const fastgltf::PlaneShape &s) -> PhysicsShape {
                        return physics::Plane{float(s.sizeX), float(s.sizeZ), s.doubleSided};
                    },
                },
                asset.shapes[*geom.shape]
            );
        }
        if (geom.convexHull) return physics::ConvexHull{};
        return physics::TriangleMesh{};
    };

    std::vector<bool> is_joint(asset.nodes.size(), false);
    for (uint32_t skin_index = 0; skin_index < asset.skins.size(); ++skin_index) {
        if (!used_skin[skin_index]) continue;
        const auto &skin = asset.skins[skin_index];
        for (const auto joint_idx : skin.joints) {
            if (const auto joint = ToIndex(joint_idx, asset.nodes.size())) is_joint[*joint] = true;
        }
    }

    // Load every source mesh in index order, so scene.Meshes[i] aligns with asset.meshes[i].
    // This preserves node.meshIndex values on round-trip. Empty-geometry meshes still get a
    // slot (Triangles/Lines/Points all nullopt), and node traversal below checks for geometry
    // presence before pointing a Node at one.
    std::unordered_map<uint32_t, uint32_t> mesh_index_map;
    scene.Meshes.reserve(asset.meshes.size());
    for (uint32_t source_mesh_index = 0; source_mesh_index < asset.meshes.size(); ++source_mesh_index) {
        auto ensured = EnsureMeshData(asset, source_mesh_index, scene, mesh_index_map);
        if (!ensured) return std::unexpected{std::move(ensured.error())};
    }

    scene.Nodes.resize(asset.nodes.size());
    for (uint32_t node_index = 0; node_index < asset.nodes.size(); ++node_index) {
        const auto &source_node = asset.nodes[node_index];
        const auto source_mesh_index = ToIndex(source_node.meshIndex, asset.meshes.size());
        // Preserve the mesh ref verbatim - runtime gates rendering on InScene, not on MeshIndex,
        // so carrying a ref on out-of-scene nodes is harmless and keeps the JSON round-trip honest.
        const auto mesh_index = source_mesh_index;
        std::vector<uint32_t> children_node_indices;
        children_node_indices.reserve(source_node.children.size());
        for (const auto child_idx : source_node.children) {
            if (const auto child = ToIndex(child_idx, asset.nodes.size())) children_node_indices.emplace_back(*child);
        }
        auto &node = scene.Nodes[node_index];
        node = Node{
            .NodeIndex = node_index,
            .ParentNodeIndex = parents[node_index],
            .ChildrenNodeIndices = std::move(children_node_indices),
            .LocalTransform = local_transforms[node_index],
            .WorldTransform = traversal.InScene[node_index] ? ToTransform(traversal.WorldTransforms[node_index]) : Transform{},
            .InScene = traversal.InScene[node_index],
            .IsJoint = is_joint[node_index],
            .SourceMatrix = source_matrices[node_index],
            .MeshIndex = mesh_index,
            .SkinIndex = ToIndex(source_node.skinIndex, asset.skins.size()),
            .CameraIndex = ToIndex(source_node.cameraIndex, asset.cameras.size()),
            .LightIndex = ToIndex(source_node.lightIndex, asset.lights.size()),
            .Name = std::string(source_node.name),
        };

        // KHR_physics_rigid_bodies per-node data
        if (const auto &rb = source_node.physicsRigidBody) {
            if (rb->motion) {
                const auto com = ToVec3(rb->motion->centerOfMass);
                // glTF quat: [x,y,z,w]
                const auto inertia_orientation = rb->motion->inertialOrientation ? std::optional{[&] { const auto q = ToVec4(*rb->motion->inertialOrientation); return quat{q.w, q.x, q.y, q.z}; }()} : std::nullopt;
                node.Motion = PhysicsMotion{
                    .IsKinematic = rb->motion->isKinematic,
                    .Mass = rb->motion->mass ? std::optional{float(*rb->motion->mass)} : std::nullopt,
                    .CenterOfMass = com != vec3{0} ? std::optional{com} : std::nullopt,
                    .InertiaDiagonal = rb->motion->inertialDiagonal ? std::optional{ToVec3(*rb->motion->inertialDiagonal)} : std::nullopt,
                    .InertiaOrientation = inertia_orientation,
                    .GravityFactor = float(rb->motion->gravityFactor),
                };
                if (const auto lv = ToVec3(rb->motion->linearVelocity), av = ToVec3(rb->motion->angularVelocity); lv != vec3{0} || av != vec3{0}) {
                    node.Velocity = {lv, av};
                }
            }
            if (rb->collider) {
                node.Collider = ColliderShape{ToPhysicsShape(rb->collider->geometry)};
                const Node::MaterialRefs material{
                    .PhysicsMaterialIndex = ToIndex(rb->collider->physicsMaterial, asset.physicsMaterials.size()),
                    .CollisionFilterIndex = ToIndex(rb->collider->collisionFilter, asset.collisionFilters.size()),
                };
                if (material.PhysicsMaterialIndex || material.CollisionFilterIndex) node.Material = material;
                // scene.Meshes is index-aligned with asset.meshes, so the glTF mesh index is also the scene mesh index.
                node.ColliderGeometryMeshIndex = ToIndex(rb->collider->geometry.mesh, asset.meshes.size());
            }
            if (rb->trigger) {
                Node::TriggerData trigger;
                std::visit(
                    [&](const auto &t) {
                        using T = std::decay_t<decltype(t)>;
                        if constexpr (std::is_same_v<T, fastgltf::GeometryTrigger>) {
                            trigger.Shape = ToPhysicsShape(t.geometry);
                            trigger.GeometryMeshIndex = ToIndex(t.geometry.mesh, asset.meshes.size());
                            trigger.CollisionFilterIndex = ToIndex(t.collisionFilter, asset.collisionFilters.size());
                        } else {
                            for (const auto n : t.nodes) {
                                if (n < asset.nodes.size()) trigger.NodeIndices.emplace_back(uint32_t(n));
                            }
                        }
                    },
                    *rb->trigger
                );
                node.Trigger = std::move(trigger);
            }
            if (rb->joint) {
                node.Joint = Node::JointData{
                    .ConnectedNodeIndex = uint32_t(rb->joint->connectedNode),
                    .JointDefIndex = uint32_t(rb->joint->joint),
                    .EnableCollision = rb->joint->enableCollision,
                };
            }
        }
    }

    std::vector<bool> is_object_emitted(asset.nodes.size(), false);
    for (uint32_t node_index = 0; node_index < scene.Nodes.size(); ++node_index) {
        if (const auto &node = scene.Nodes[node_index]; node.InScene) {
            // Joint nodes are bone-only unless they also carry renderable mesh data.
            is_object_emitted[node_index] = node.MeshIndex.has_value() || !node.IsJoint;
        }
    }

    std::vector<std::optional<uint32_t>> nearest_object_ancestor(asset.nodes.size());
    for (uint32_t node_index = 0; node_index < asset.nodes.size(); ++node_index) {
        nearest_object_ancestor[node_index] = FindNearestEmittedObjectAncestor(node_index, parents, is_object_emitted);
    }

    // Precompute per-node instance transforms for EXT_mesh_gpu_instancing
    std::vector<std::vector<Transform>> node_instance_transforms(asset.nodes.size());
    for (uint32_t ni = 0; ni < asset.nodes.size(); ++ni) {
        if (traversal.InScene[ni]) node_instance_transforms[ni] = ReadInstanceTransforms(asset, asset.nodes[ni]);
    }

    for (uint32_t node_index = 0; node_index < scene.Nodes.size(); ++node_index) {
        if (const auto &node = scene.Nodes[node_index]; is_object_emitted[node_index]) {
            const auto source_mesh_index = ToIndex(asset.nodes[node_index].meshIndex, asset.meshes.size());
            if (const auto &instance_transforms = node_instance_transforms[node_index];
                !instance_transforms.empty() && node.MeshIndex) {
                // EXT_mesh_gpu_instancing: emit one object per instance with baked world transform
                const auto base_name = MakeNodeName(asset, node.NodeIndex, source_mesh_index);
                const auto &source_weights = asset.nodes[node_index].weights;
                auto node_weights = source_weights.empty() ? std::optional<std::vector<float>>{} : std::optional{std::vector<float>(source_weights.begin(), source_weights.end())};
                for (uint32_t i = 0; i < instance_transforms.size(); ++i) {
                    scene.Objects.emplace_back(Object{
                        .ObjectType = Object::Type::Mesh,
                        .NodeIndex = node.NodeIndex,
                        .ParentNodeIndex = std::nullopt,
                        .WorldTransform = ToTransform(traversal.WorldTransforms[node_index] * ToMatrix(instance_transforms[i])),
                        .MeshIndex = node.MeshIndex,
                        .SkinIndex = node.SkinIndex,
                        .CameraIndex = {},
                        .LightIndex = {},
                        .NodeWeights = node_weights,
                        .Name = base_name + "." + std::to_string(i),
                    });
                }
            } else {
                const auto &source_weights = asset.nodes[node_index].weights;
                const auto object_type = node.MeshIndex ? Object::Type::Mesh :
                    node.CameraIndex                    ? Object::Type::Camera :
                    node.LightIndex                     ? Object::Type::Light :
                                                          Object::Type::Empty;
                scene.Objects.emplace_back(Object{
                    .ObjectType = object_type,
                    .NodeIndex = node.NodeIndex,
                    .ParentNodeIndex = nearest_object_ancestor[node.NodeIndex],
                    .WorldTransform = node.WorldTransform,
                    .MeshIndex = node.MeshIndex,
                    .SkinIndex = node.SkinIndex,
                    .CameraIndex = node.CameraIndex,
                    .LightIndex = node.LightIndex,
                    .NodeWeights = source_weights.empty() ? std::optional<std::vector<float>>{} : std::optional{std::vector<float>(source_weights.begin(), source_weights.end())},
                    .Name = MakeNodeName(asset, node.NodeIndex, source_mesh_index),
                });
            }
        }
    }

    scene.Skins.reserve(asset.skins.size());
    for (uint32_t skin_index = 0; skin_index < asset.skins.size(); ++skin_index) {
        if (!used_skin[skin_index]) continue;
        const auto &skin = asset.skins[skin_index];

        std::vector<uint32_t> source_joint_nodes;
        source_joint_nodes.reserve(skin.joints.size());
        std::unordered_set<uint32_t> seen_joints;
        for (const auto joint_idx : skin.joints) {
            if (const auto joint = ToIndex(joint_idx, asset.nodes.size())) {
                if (const auto [it, inserted] = seen_joints.emplace(*joint); inserted) {
                    source_joint_nodes.emplace_back(*it);
                }
            }
        }
        if (source_joint_nodes.empty()) continue;

        std::vector<bool> is_joint_in_skin(asset.nodes.size(), false);
        for (const auto joint_node_index : source_joint_nodes) is_joint_in_skin[joint_node_index] = true;

        std::unordered_map<uint32_t, std::optional<uint32_t>> joint_parent_map;
        joint_parent_map.reserve(source_joint_nodes.size());
        for (const auto joint_node_index : source_joint_nodes) {
            joint_parent_map.emplace(joint_node_index, FindNearestJointAncestor(joint_node_index, parents, is_joint_in_skin));
        }

        auto ordered_joint_nodes = BuildParentBeforeChildJointOrder(source_joint_nodes, joint_parent_map, skin_index);
        if (!ordered_joint_nodes) return std::unexpected{ordered_joint_nodes.error()};
        // AddBone/FinalizeStructure consume joints in this parent-before-child order.

        const auto skeleton_node_index = ToIndex(skin.skeleton, asset.nodes.size());
        // Deterministic armature scene anchor: explicit skin.skeleton if present,
        // otherwise the computed joint ancestry root. Do not synthesize extra roots.
        Skin scene_skin{
            .SkinIndex = skin_index,
            .Name = std::string(skin.name),
            .SkeletonNodeIndex = skeleton_node_index,
            .AnchorNodeIndex = skeleton_node_index ? skeleton_node_index : ComputeCommonAncestor(*ordered_joint_nodes, parents),
            .Joints = {},
            .InverseBindMatrices = {},
        };
        scene_skin.Joints.reserve(ordered_joint_nodes->size());
        for (const auto joint_node_index : *ordered_joint_nodes) {
            const auto parent_joint_node_index = joint_parent_map.at(joint_node_index);
            auto rest_local = ComputeJointRestLocal(skin_index, joint_node_index, parent_joint_node_index, scene_skin.AnchorNodeIndex, parents, local_transforms);
            if (!rest_local) return std::unexpected{rest_local.error()};

            scene_skin.Joints.emplace_back(SkinJoint{
                .JointNodeIndex = joint_node_index,
                .ParentJointNodeIndex = parent_joint_node_index,
                .RestLocal = *rest_local,
                .Name = MakeNodeName(asset, joint_node_index),
            });
        }
        scene_skin.InverseBindMatrices = LoadInverseBindMatrices(asset, skin, uint32_t(scene_skin.Joints.size()));
        if (scene_skin.AnchorNodeIndex && traversal.InScene[*scene_skin.AnchorNodeIndex]) {
            scene_skin.ParentObjectNodeIndex = nearest_object_ancestor[*scene_skin.AnchorNodeIndex];
        }
        scene.Skins.emplace_back(std::move(scene_skin));
    }

    // Parse animations
    for (uint32_t anim_index = 0; anim_index < asset.animations.size(); ++anim_index) {
        const auto &anim = asset.animations[anim_index];
        std::vector<AnimationChannel> channels;
        float max_time = 0;
        struct ChannelTargetSpec {
            AnimationPath Path;
            std::size_t ComponentCount;
        };
        for (const auto &channel : anim.channels) {
            if (!channel.nodeIndex || *channel.nodeIndex >= asset.nodes.size()) continue;
            if (channel.samplerIndex >= anim.samplers.size()) continue;

            const auto target_spec = [&]() -> std::optional<ChannelTargetSpec> {
                switch (channel.path) {
                    case fastgltf::AnimationPath::Translation: return ChannelTargetSpec{.Path = AnimationPath::Translation, .ComponentCount = 3};
                    case fastgltf::AnimationPath::Rotation: return ChannelTargetSpec{.Path = AnimationPath::Rotation, .ComponentCount = 4};
                    case fastgltf::AnimationPath::Scale: return ChannelTargetSpec{.Path = AnimationPath::Scale, .ComponentCount = 3};
                    case fastgltf::AnimationPath::Weights: {
                        const auto &target_node = asset.nodes[*channel.nodeIndex];
                        if (!target_node.meshIndex || *target_node.meshIndex >= asset.meshes.size()) return std::nullopt;
                        const auto component_count = asset.meshes[*target_node.meshIndex].primitives.empty() ? 0 : asset.meshes[*target_node.meshIndex].primitives[0].targets.size();
                        if (component_count == 0) return std::nullopt;
                        return ChannelTargetSpec{.Path = AnimationPath::Weights, .ComponentCount = component_count};
                    }
                }
                return std::nullopt;
            }();
            if (!target_spec) continue;

            const auto &sampler = anim.samplers[channel.samplerIndex];
            if (sampler.inputAccessor >= asset.accessors.size() || sampler.outputAccessor >= asset.accessors.size()) continue;

            const auto &input_accessor = asset.accessors[sampler.inputAccessor];
            const auto &output_accessor = asset.accessors[sampler.outputAccessor];
            if (input_accessor.count == 0) continue;

            const auto interp = [&]() {
                switch (sampler.interpolation) {
                    case fastgltf::AnimationInterpolation::Step: return AnimationInterpolation::Step;
                    case fastgltf::AnimationInterpolation::Linear: return AnimationInterpolation::Linear;
                    case fastgltf::AnimationInterpolation::CubicSpline: return AnimationInterpolation::CubicSpline;
                }
                return AnimationInterpolation::Linear;
            }();

            std::vector<float> times(input_accessor.count);
            fastgltf::copyFromAccessor<float>(asset, input_accessor, times.data());

            std::vector<float> values;
            if (target_spec->Path == AnimationPath::Weights) {
                // Output is keyframe_count * target_count scalars.
                values.resize(output_accessor.count);
                fastgltf::copyFromAccessor<float>(asset, output_accessor, values.data());
            } else {
                values.resize(output_accessor.count * target_spec->ComponentCount);
                if (target_spec->ComponentCount == 4) fastgltf::copyFromAccessor<vec4>(asset, output_accessor, reinterpret_cast<vec4 *>(values.data()));
                else fastgltf::copyFromAccessor<vec3>(asset, output_accessor, reinterpret_cast<vec3 *>(values.data()));
            }

            if (!times.empty()) max_time = std::max(max_time, times.back());

            channels.emplace_back(AnimationChannel{.TargetNodeIndex = uint32_t(*channel.nodeIndex), .Target = target_spec->Path, .Interp = interp, .TimesSeconds = std::move(times), .Values = std::move(values)});
        }

        if (!channels.empty()) {
            scene.Animations.emplace_back(AnimationClip{
                .Name = std::string(anim.name),
                .DurationSeconds = max_time,
                .Channels = std::move(channels),
            });
        }
    }

    if (scene.Objects.empty() && scene.Skins.empty()) {
        return std::unexpected{std::format("glTF '{}' has no importable scene objects or skins.", path.string())};
    }
    return scene;
}

namespace {
fastgltf::Filter FromFilter(Filter f) {
    switch (f) {
        case Filter::Nearest: return fastgltf::Filter::Nearest;
        case Filter::Linear: return fastgltf::Filter::Linear;
        case Filter::NearestMipMapNearest: return fastgltf::Filter::NearestMipMapNearest;
        case Filter::LinearMipMapNearest: return fastgltf::Filter::LinearMipMapNearest;
        case Filter::NearestMipMapLinear: return fastgltf::Filter::NearestMipMapLinear;
        case Filter::LinearMipMapLinear: return fastgltf::Filter::LinearMipMapLinear;
    }
    return fastgltf::Filter::LinearMipMapLinear;
}
fastgltf::Wrap FromWrap(Wrap w) {
    switch (w) {
        case Wrap::ClampToEdge: return fastgltf::Wrap::ClampToEdge;
        case Wrap::MirroredRepeat: return fastgltf::Wrap::MirroredRepeat;
        case Wrap::Repeat: return fastgltf::Wrap::Repeat;
    }
    return fastgltf::Wrap::Repeat;
}
fastgltf::MimeType FromMimeType(MimeType m) {
    switch (m) {
        case MimeType::None: return fastgltf::MimeType::None;
        case MimeType::JPEG: return fastgltf::MimeType::JPEG;
        case MimeType::PNG: return fastgltf::MimeType::PNG;
        case MimeType::KTX2: return fastgltf::MimeType::KTX2;
        case MimeType::DDS: return fastgltf::MimeType::DDS;
        case MimeType::GltfBuffer: return fastgltf::MimeType::GltfBuffer;
        case MimeType::OctetStream: return fastgltf::MimeType::OctetStream;
        case MimeType::WEBP: return fastgltf::MimeType::WEBP;
    }
    return fastgltf::MimeType::None;
}
fastgltf::AnimationInterpolation FromInterp(AnimationInterpolation i) {
    switch (i) {
        case AnimationInterpolation::Step: return fastgltf::AnimationInterpolation::Step;
        case AnimationInterpolation::Linear: return fastgltf::AnimationInterpolation::Linear;
        case AnimationInterpolation::CubicSpline: return fastgltf::AnimationInterpolation::CubicSpline;
    }
    return fastgltf::AnimationInterpolation::Linear;
}
fastgltf::AnimationPath FromPath(AnimationPath p) {
    switch (p) {
        case AnimationPath::Translation: return fastgltf::AnimationPath::Translation;
        case AnimationPath::Rotation: return fastgltf::AnimationPath::Rotation;
        case AnimationPath::Scale: return fastgltf::AnimationPath::Scale;
        case AnimationPath::Weights: return fastgltf::AnimationPath::Weights;
    }
    return fastgltf::AnimationPath::Translation;
}
fastgltf::CombineMode FromCombine(PhysicsCombineMode m) {
    switch (m) {
        case PhysicsCombineMode::Average: return fastgltf::CombineMode::Average;
        case PhysicsCombineMode::Minimum: return fastgltf::CombineMode::Minimum;
        case PhysicsCombineMode::Maximum: return fastgltf::CombineMode::Maximum;
        case PhysicsCombineMode::Multiply: return fastgltf::CombineMode::Multiply;
    }
    return fastgltf::CombineMode::Average;
}

// fastgltf's name fields are pmr::string (doesn't implicit-copy from std::string).
using FgString = std::remove_cvref_t<decltype(fastgltf::Material::name)>;
FgString ToFgStr(std::string_view s) { return FgString{s}; }

// fastgltf::Optional<T> doesn't implicit-convert from std::optional<U>.
template<typename T, typename U>
fastgltf::Optional<T> ToFgOpt(const std::optional<U> &o) {
    return o ? fastgltf::Optional<T>{T(*o)} : fastgltf::Optional<T>{};
}
template<typename T, typename U, typename Fn>
fastgltf::Optional<T> ToFgOpt(const std::optional<U> &o, Fn &&fn) {
    return o ? fastgltf::Optional<T>{fn(*o)} : fastgltf::Optional<T>{};
}

std::optional<fastgltf::AccessorBoundsArray> MakeBounds(std::initializer_list<double> vals) {
    auto arr = fastgltf::AccessorBoundsArray::ForType<double>(vals.size());
    std::size_t i = 0;
    for (const double v : vals) arr.set<double>(i++, v);
    return arr;
}

// Append bytes, pad to 4-byte alignment, return starting offset.
uint32_t AppendAligned(std::vector<std::byte> &buffer, const std::byte *data, uint32_t size) {
    const uint32_t offset = buffer.size();
    buffer.insert(buffer.end(), data, data + size);
    while (buffer.size() % 4 != 0) buffer.emplace_back(std::byte{0});
    return offset;
}

template<typename T>
uint32_t AppendAligned(std::vector<std::byte> &buffer, std::span<const T> data) {
    return AppendAligned(buffer, reinterpret_cast<const std::byte *>(data.data()), uint32_t(data.size() * sizeof(T)));
}

fastgltf::AlphaMode FromAlphaMode(MaterialAlphaMode m) {
    switch (m) {
        case MaterialAlphaMode::Opaque: return fastgltf::AlphaMode::Opaque;
        case MaterialAlphaMode::Mask: return fastgltf::AlphaMode::Mask;
        case MaterialAlphaMode::Blend: return fastgltf::AlphaMode::Blend;
    }
    return fastgltf::AlphaMode::Opaque;
}

std::unique_ptr<fastgltf::TextureTransform> MakeTextureTransform(const ::TextureInfo &ti, const TextureTransformMeta *meta = nullptr) {
    const bool has_transform = ti.UvOffset.x != 0.f || ti.UvOffset.y != 0.f ||
        ti.UvScale.x != 1.f || ti.UvScale.y != 1.f ||
        ti.UvRotation != 0.f;
    const bool source_had_ext = meta && meta->SourceHadExtension;
    if (!has_transform && !source_had_ext) return nullptr;
    auto t = std::make_unique<fastgltf::TextureTransform>();
    t->rotation = ti.UvRotation;
    t->uvOffset = {ti.UvOffset.x, ti.UvOffset.y};
    t->uvScale = {ti.UvScale.x, ti.UvScale.y};
    if (meta && meta->SourceTexCoordOverride) t->texCoordIndex = *meta->SourceTexCoordOverride;
    return t;
}

void FillFgTextureInfo(fastgltf::TextureInfo &out, const ::TextureInfo &ti, const TextureTransformMeta *meta = nullptr) {
    out.textureIndex = ti.Slot;
    // With meta, the parent texCoord and the extension's override are emitted separately.
    out.texCoordIndex = meta ? meta->SourceBaseTexCoord : ti.TexCoord;
    out.transform = MakeTextureTransform(ti, meta);
}

fastgltf::Optional<fastgltf::TextureInfo> ToFgTexInfo(const ::TextureInfo &ti, const TextureTransformMeta *meta = nullptr) {
    if (ti.Slot == InvalidSlot) return {};
    fastgltf::TextureInfo out;
    FillFgTextureInfo(out, ti, meta);
    return fastgltf::Optional<fastgltf::TextureInfo>{std::move(out)};
}

fastgltf::Optional<fastgltf::NormalTextureInfo> ToFgNormalTexInfo(const ::TextureInfo &ti, float scale, const TextureTransformMeta *meta = nullptr) {
    if (ti.Slot == InvalidSlot) return {};
    fastgltf::NormalTextureInfo out;
    FillFgTextureInfo(out, ti, meta);
    out.scale = scale;
    return fastgltf::Optional<fastgltf::NormalTextureInfo>{std::move(out)};
}

fastgltf::Optional<fastgltf::OcclusionTextureInfo> ToFgOcclusionTexInfo(const ::TextureInfo &ti, float strength, const TextureTransformMeta *meta = nullptr) {
    if (ti.Slot == InvalidSlot) return {};
    fastgltf::OcclusionTextureInfo out;
    FillFgTextureInfo(out, ti, meta);
    out.strength = strength;
    return fastgltf::Optional<fastgltf::OcclusionTextureInfo>{std::move(out)};
}

} // namespace

std::expected<void, std::string> SaveScene(const Scene &scene, const std::filesystem::path &path) {
    fastgltf::Asset asset;
    // Empty strings are omitted by fastgltf's writer.
    asset.assetInfo = fastgltf::AssetInfo{
        .gltfVersion = "2.0",
        .minVersion = ToFgStr(scene.MinVersion),
        .copyright = ToFgStr(scene.Copyright),
        .generator = ToFgStr(scene.Generator),
        .extras = ToFgStr(scene.AssetExtras),
        .extensions = ToFgStr(scene.AssetExtensions),
    };

    std::vector<std::byte> bin;
    std::vector<fastgltf::BufferView> bufferViews;
    std::vector<fastgltf::Accessor> accessors;

    auto AddBufferView = [&](uint32_t offset, uint32_t length, std::optional<uint32_t> stride = {}, std::optional<fastgltf::BufferTarget> target = {}) {
        bufferViews.emplace_back(fastgltf::BufferView{
            .bufferIndex = 0,
            .byteOffset = offset,
            .byteLength = length,
            .byteStride = ToFgOpt<std::size_t>(stride),
            .target = ToFgOpt<fastgltf::BufferTarget>(target),
            .meshoptCompression = nullptr,
            .name = {},
        });
        return uint32_t(bufferViews.size() - 1);
    };

    auto AddAccessor = [&](uint32_t bufferViewIdx, uint32_t count, fastgltf::AccessorType type, fastgltf::ComponentType component,
                           std::optional<fastgltf::AccessorBoundsArray> min = {}, std::optional<fastgltf::AccessorBoundsArray> max = {}) {
        accessors.emplace_back(fastgltf::Accessor{
            .byteOffset = 0,
            .count = count,
            .type = type,
            .componentType = component,
            .normalized = false,
            .max = std::move(max),
            .min = std::move(min),
            .bufferViewIndex = bufferViewIdx,
            .sparse = {},
            .name = {},
        });
        return uint32_t(accessors.size() - 1);
    };

    const auto AddDataAccessor = [&]<typename T>(std::span<const T> data, fastgltf::AccessorType type, fastgltf::ComponentType component, std::optional<fastgltf::BufferTarget> target = {}) {
        const uint32_t off = AppendAligned<T>(bin, data);
        const uint32_t bv = AddBufferView(off, uint32_t(data.size() * sizeof(T)), {}, target);
        return AddAccessor(bv, uint32_t(data.size()), type, component);
    };

    const auto AddVec3Accessor = [&](std::span<const vec3> data, bool with_bounds, fastgltf::BufferTarget target) {
        const uint32_t vcount = uint32_t(data.size());
        if (!with_bounds || vcount == 0) return AddDataAccessor(data, fastgltf::AccessorType::Vec3, fastgltf::ComponentType::Float, target);
        vec3 lo = data[0], hi = data[0];
        for (const auto &p : data) {
            lo = glm::min(lo, p);
            hi = glm::max(hi, p);
        }
        const uint32_t off = AppendAligned<vec3>(bin, data);
        const uint32_t bv = AddBufferView(off, vcount * sizeof(vec3), {}, target);
        return AddAccessor(
            bv, vcount, fastgltf::AccessorType::Vec3, fastgltf::ComponentType::Float,
            MakeBounds({lo.x, lo.y, lo.z}), MakeBounds({hi.x, hi.y, hi.z})
        );
    };

    // Emit COLOR_0 preserving source component count (vec3 narrows the vec4 CPU store; vec4 passes through).
    const auto EmitColor0 = [&](fastgltf::pmr::SmallVector<fastgltf::Attribute, 4> &out, std::span<const vec4> src, uint8_t component_count) {
        if (component_count == 3) {
            std::vector<vec3> rgb(src.size());
            for (std::size_t i = 0; i < src.size(); ++i) rgb[i] = vec3{src[i].x, src[i].y, src[i].z};
            out.emplace_back(fastgltf::Attribute{"COLOR_0", AddVec3Accessor(rgb, false, fastgltf::BufferTarget::ArrayBuffer)});
        } else {
            const uint32_t acc = AddDataAccessor(src, fastgltf::AccessorType::Vec4, fastgltf::ComponentType::Float, fastgltf::BufferTarget::ArrayBuffer);
            out.emplace_back(fastgltf::Attribute{"COLOR_0", acc});
        }
    };

    // Samplers
    asset.samplers.reserve(scene.Samplers.size());
    for (const auto &s : scene.Samplers) {
        asset.samplers.emplace_back(fastgltf::Sampler{
            .magFilter = ToFgOpt<fastgltf::Filter>(s.MagFilter, FromFilter),
            .minFilter = ToFgOpt<fastgltf::Filter>(s.MinFilter, FromFilter),
            .wrapS = FromWrap(s.WrapS),
            .wrapT = FromWrap(s.WrapT),
            .name = ToFgStr(s.Name),
        });
    }

    // Images: re-emit in the source's form - data URI, external URI, or embedded bufferView.
    std::vector<uint32_t> image_bv_index(scene.Images.size(), UINT32_MAX);
    asset.images.reserve(scene.Images.size());
    for (uint32_t i = 0; i < scene.Images.size(); ++i) {
        const auto &img = scene.Images[i];
        if (img.SourceDataUri && !img.Bytes.empty()) {
            // The data URI carries the mime; suppress the separate mimeType field (source didn't have one).
            const auto fg_mime = FromMimeType(img.MimeType);
            const auto mime_str = fg_mime == fastgltf::MimeType::None ? std::string{} : std::string(fastgltf::getMimeTypeString(fg_mime));
            const auto data_uri = "data:" + mime_str + ";base64," + fastgltf::base64::encode(reinterpret_cast<const std::uint8_t *>(img.Bytes.data()), img.Bytes.size());
            asset.images.emplace_back(fastgltf::Image{
                .data = fastgltf::sources::URI{.fileByteOffset = 0, .uri = fastgltf::URI{data_uri}, .mimeType = fastgltf::MimeType::None},
                .name = ToFgStr(img.Name),
            });
            continue;
        }
        if (!img.Uri.empty()) {
            // mimeType is optional for URI-form; our magic-byte inference (runtime-only) shouldn't leak to output.
            const auto emit_mime = img.SourceHadMimeType ? FromMimeType(img.MimeType) : fastgltf::MimeType::None;
            asset.images.emplace_back(fastgltf::Image{
                .data = fastgltf::sources::URI{.fileByteOffset = 0, .uri = fastgltf::URI{std::string_view{img.Uri}}, .mimeType = emit_mime},
                .name = ToFgStr(img.Name),
            });
            continue;
        }
        const uint32_t bv = [&] {
            if (!img.Bytes.empty()) {
                const uint32_t offset = AppendAligned(bin, img.Bytes.data(), uint32_t(img.Bytes.size()));
                return AddBufferView(offset, uint32_t(img.Bytes.size()));
            }
            // Placeholder empty bufferView.
            const uint32_t offset = bin.size();
            bin.emplace_back(std::byte{0});
            while (bin.size() % 4 != 0) bin.emplace_back(std::byte{0});
            return AddBufferView(offset, 1);
        }();
        image_bv_index[i] = bv;
        asset.images.emplace_back(fastgltf::Image{
            .data = fastgltf::sources::BufferView{.bufferViewIndex = bv, .mimeType = FromMimeType(img.MimeType)},
            .name = ToFgStr(img.Name),
        });
    }

    // Textures
    asset.textures.reserve(scene.Textures.size());
    for (const auto &t : scene.Textures) {
        asset.textures.emplace_back(fastgltf::Texture{
            .samplerIndex = ToFgOpt<std::size_t>(t.SamplerIndex),
            .imageIndex = ToFgOpt<std::size_t>(t.ImageIndex),
            .basisuImageIndex = ToFgOpt<std::size_t>(t.BasisuImageIndex),
            .ddsImageIndex = ToFgOpt<std::size_t>(t.DdsImageIndex),
            .webpImageIndex = ToFgOpt<std::size_t>(t.WebpImageIndex),
            .name = ToFgStr(t.Name),
        });
    }

    // Materials: skip the trailing synthetic "DefaultMaterial" the loader appends.
    const uint32_t material_count = scene.Materials.empty() ? 0 : uint32_t(scene.Materials.size() - 1);
    asset.materials.reserve(material_count);
    for (uint32_t i = 0; i < material_count; ++i) {
        const auto &m = scene.Materials[i].Value;
        fastgltf::Material out;
        out.name = ToFgStr(scene.Materials[i].Name);

        out.pbrData.baseColorFactor = {m.BaseColorFactor.x, m.BaseColorFactor.y, m.BaseColorFactor.z, m.BaseColorFactor.w};
        out.pbrData.metallicFactor = m.MetallicFactor;
        out.pbrData.roughnessFactor = m.RoughnessFactor;
        out.pbrData.baseColorTexture = ToFgTexInfo(m.BaseColorTexture, &m.BaseColorMeta);
        out.pbrData.metallicRoughnessTexture = ToFgTexInfo(m.MetallicRoughnessTexture, &m.MetallicRoughnessMeta);

        out.normalTexture = ToFgNormalTexInfo(m.NormalTexture, m.NormalScale, &m.NormalMeta);
        out.occlusionTexture = ToFgOcclusionTexInfo(m.OcclusionTexture, m.OcclusionStrength, &m.OcclusionMeta);
        out.emissiveTexture = ToFgTexInfo(m.EmissiveTexture, &m.EmissiveMeta);

        out.emissiveFactor = {m.EmissiveFactor.x, m.EmissiveFactor.y, m.EmissiveFactor.z};
        if (m.EmissiveStrength) out.emissiveStrength = fastgltf::Optional<fastgltf::num>{*m.EmissiveStrength};

        out.alphaMode = FromAlphaMode(m.AlphaMode);
        out.alphaCutoff = m.AlphaCutoff;
        out.doubleSided = m.DoubleSided != 0u;
        out.unlit = m.Unlit != 0u;
        if (m.Ior) out.ior = fastgltf::Optional<fastgltf::num>{*m.Ior};
        if (m.Dispersion) out.dispersion = fastgltf::Optional<fastgltf::num>{*m.Dispersion};

        if (m.Sheen) {
            out.sheen = std::make_unique<fastgltf::MaterialSheen>();
            out.sheen->sheenColorFactor = {m.Sheen->ColorFactor.x, m.Sheen->ColorFactor.y, m.Sheen->ColorFactor.z};
            out.sheen->sheenRoughnessFactor = m.Sheen->RoughnessFactor;
            out.sheen->sheenColorTexture = ToFgTexInfo(m.Sheen->ColorTexture);
            out.sheen->sheenRoughnessTexture = ToFgTexInfo(m.Sheen->RoughnessTexture);
        }
        if (m.Specular) {
            out.specular = std::make_unique<fastgltf::MaterialSpecular>();
            out.specular->specularFactor = m.Specular->Factor;
            out.specular->specularColorFactor = {m.Specular->ColorFactor.x, m.Specular->ColorFactor.y, m.Specular->ColorFactor.z};
            out.specular->specularTexture = ToFgTexInfo(m.Specular->Texture);
            out.specular->specularColorTexture = ToFgTexInfo(m.Specular->ColorTexture);
        }
        if (m.Transmission) {
            out.transmission = std::make_unique<fastgltf::MaterialTransmission>();
            out.transmission->transmissionFactor = m.Transmission->Factor;
            out.transmission->transmissionTexture = ToFgTexInfo(m.Transmission->Texture);
        }
        if (m.DiffuseTransmission) {
            out.diffuseTransmission = std::make_unique<fastgltf::MaterialDiffuseTransmission>();
            out.diffuseTransmission->diffuseTransmissionFactor = m.DiffuseTransmission->Factor;
            out.diffuseTransmission->diffuseTransmissionColorFactor = {m.DiffuseTransmission->ColorFactor.x, m.DiffuseTransmission->ColorFactor.y, m.DiffuseTransmission->ColorFactor.z};
            out.diffuseTransmission->diffuseTransmissionTexture = ToFgTexInfo(m.DiffuseTransmission->Texture);
            out.diffuseTransmission->diffuseTransmissionColorTexture = ToFgTexInfo(m.DiffuseTransmission->ColorTexture);
        }
        if (m.Volume) {
            out.volume = std::make_unique<fastgltf::MaterialVolume>();
            out.volume->thicknessFactor = m.Volume->ThicknessFactor;
            out.volume->attenuationColor = {m.Volume->AttenuationColor.x, m.Volume->AttenuationColor.y, m.Volume->AttenuationColor.z};
            out.volume->attenuationDistance = m.Volume->AttenuationDistance > 0.f ? m.Volume->AttenuationDistance : std::numeric_limits<float>::infinity();
            out.volume->thicknessTexture = ToFgTexInfo(m.Volume->ThicknessTexture);
        }
        if (m.Clearcoat) {
            out.clearcoat = std::make_unique<fastgltf::MaterialClearcoat>();
            out.clearcoat->clearcoatFactor = m.Clearcoat->Factor;
            out.clearcoat->clearcoatRoughnessFactor = m.Clearcoat->RoughnessFactor;
            out.clearcoat->clearcoatTexture = ToFgTexInfo(m.Clearcoat->Texture);
            out.clearcoat->clearcoatRoughnessTexture = ToFgTexInfo(m.Clearcoat->RoughnessTexture);
            out.clearcoat->clearcoatNormalTexture = ToFgNormalTexInfo(m.Clearcoat->NormalTexture, m.Clearcoat->NormalScale);
        }
        if (m.Anisotropy) {
            out.anisotropy = std::make_unique<fastgltf::MaterialAnisotropy>();
            out.anisotropy->anisotropyStrength = m.Anisotropy->Strength;
            out.anisotropy->anisotropyRotation = m.Anisotropy->Rotation;
            out.anisotropy->anisotropyTexture = ToFgTexInfo(m.Anisotropy->Texture);
        }
        if (m.Iridescence) {
            out.iridescence = std::make_unique<fastgltf::MaterialIridescence>();
            out.iridescence->iridescenceFactor = m.Iridescence->Factor;
            out.iridescence->iridescenceIor = m.Iridescence->Ior;
            out.iridescence->iridescenceThicknessMinimum = m.Iridescence->ThicknessMinimum;
            out.iridescence->iridescenceThicknessMaximum = m.Iridescence->ThicknessMaximum;
            out.iridescence->iridescenceTexture = ToFgTexInfo(m.Iridescence->Texture);
            out.iridescence->iridescenceThicknessTexture = ToFgTexInfo(m.Iridescence->ThicknessTexture);
        }

        asset.materials.emplace_back(std::move(out));
    }

    asset.meshes.reserve(scene.Meshes.size());
    for (uint32_t mi = 0; mi < scene.Meshes.size(); ++mi) {
        const auto &mesh = scene.Meshes[mi];
        fastgltf::pmr::MaybeSmallVector<fastgltf::Primitive, 2> primitives;

        // --- Triangle primitives ---
        // Each source primitive owns a contiguous [offset, offset+count) slice of the merged
        // per-mesh vertex buffer; re-slice here by primitive range.
        if (mesh.Triangles && !mesh.Triangles->Positions.empty() && !mesh.Triangles->Faces.empty()) {
            const auto &tri = *mesh.Triangles;
            const uint32_t total_vcount = uint32_t(tri.Positions.size());
            const auto &tprims = mesh.TrianglePrimitives;

            std::vector<uint32_t> vertex_offsets(tprims.VertexCounts.size() + 1, 0);
            for (std::size_t p = 0; p < tprims.VertexCounts.size(); ++p) vertex_offsets[p + 1] = vertex_offsets[p] + tprims.VertexCounts[p];

            std::unordered_map<uint32_t, std::vector<uint32_t>> faces_by_prim;
            for (uint32_t fi = 0; fi < tri.Faces.size(); ++fi) {
                const uint32_t p = fi < tprims.FacePrimitiveIndices.size() ? tprims.FacePrimitiveIndices[fi] : 0u;
                faces_by_prim[p].emplace_back(fi);
            }

            const auto &attrs = mesh.TriangleAttrs;
            const bool has_normals = attrs.Normals && attrs.Normals->size() == total_vcount;
            const bool has_tangents = attrs.Tangents && attrs.Tangents->size() == total_vcount;
            const bool has_colors = attrs.Colors0 && attrs.Colors0->size() == total_vcount;
            const bool has_uv0 = attrs.TexCoords0 && attrs.TexCoords0->size() == total_vcount;
            const bool has_uv1 = attrs.TexCoords1 && attrs.TexCoords1->size() == total_vcount;
            const bool has_uv2 = attrs.TexCoords2 && attrs.TexCoords2->size() == total_vcount;
            const bool has_uv3 = attrs.TexCoords3 && attrs.TexCoords3->size() == total_vcount;
            const bool has_skin = mesh.DeformData && mesh.DeformData->Joints.size() == total_vcount && mesh.DeformData->Weights.size() == total_vcount;
            const uint32_t target_count = mesh.MorphData ? mesh.MorphData->TargetCount : 0;
            const bool has_normal_deltas = mesh.MorphData && !mesh.MorphData->NormalDeltas.empty();
            const bool has_tangent_deltas = mesh.MorphData && !mesh.MorphData->TangentDeltas.empty();
            // Fall back to emitting every populated channel when AttributeFlags isn't populated.
            const bool have_flags = tprims.AttributeFlags.size() == tprims.VertexCounts.size();

            for (uint32_t prim_idx = 0; prim_idx < tprims.VertexCounts.size(); ++prim_idx) {
                const uint32_t pcount = tprims.VertexCounts[prim_idx];
                if (pcount == 0) continue; // non-triangle primitive
                const uint32_t offset = vertex_offsets[prim_idx];
                const uint32_t flags = have_flags ? tprims.AttributeFlags[prim_idx] : ~0u;

                fastgltf::pmr::SmallVector<fastgltf::Attribute, 4> prim_attrs;

                const std::span<const vec3> pos_span(tri.Positions.data() + offset, pcount);
                prim_attrs.emplace_back(fastgltf::Attribute{"POSITION", AddVec3Accessor(pos_span, true, fastgltf::BufferTarget::ArrayBuffer)});

                if (has_normals && (flags & MeshAttributeBit_Normal)) {
                    const std::span<const vec3> norm_span(attrs.Normals->data() + offset, pcount);
                    prim_attrs.emplace_back(fastgltf::Attribute{"NORMAL", AddVec3Accessor(norm_span, false, fastgltf::BufferTarget::ArrayBuffer)});
                }

                const auto emit_vec4 = [&](const char *name, const vec4 *src) {
                    const uint32_t acc = AddDataAccessor(std::span<const vec4>(src, pcount), fastgltf::AccessorType::Vec4, fastgltf::ComponentType::Float, fastgltf::BufferTarget::ArrayBuffer);
                    prim_attrs.emplace_back(fastgltf::Attribute{name, acc});
                };
                if (has_tangents && (flags & MeshAttributeBit_Tangent)) emit_vec4("TANGENT", attrs.Tangents->data() + offset);
                if (has_colors && (flags & MeshAttributeBit_Color0)) {
                    EmitColor0(prim_attrs, std::span<const vec4>(attrs.Colors0->data() + offset, pcount), attrs.Colors0ComponentCount);
                }

                const auto emit_uv = [&](const char *name, const vec2 *src) {
                    const uint32_t acc = AddDataAccessor(std::span<const vec2>(src, pcount), fastgltf::AccessorType::Vec2, fastgltf::ComponentType::Float, fastgltf::BufferTarget::ArrayBuffer);
                    prim_attrs.emplace_back(fastgltf::Attribute{name, acc});
                };
                if (has_uv0 && (flags & MeshAttributeBit_TexCoord0)) emit_uv("TEXCOORD_0", attrs.TexCoords0->data() + offset);
                if (has_uv1 && (flags & MeshAttributeBit_TexCoord1)) emit_uv("TEXCOORD_1", attrs.TexCoords1->data() + offset);
                if (has_uv2 && (flags & MeshAttributeBit_TexCoord2)) emit_uv("TEXCOORD_2", attrs.TexCoords2->data() + offset);
                if (has_uv3 && (flags & MeshAttributeBit_TexCoord3)) emit_uv("TEXCOORD_3", attrs.TexCoords3->data() + offset);

                if (has_skin) {
                    std::vector<std::array<uint16_t, 4>> joints16(pcount);
                    for (uint32_t i = 0; i < pcount; ++i) {
                        const auto &j = mesh.DeformData->Joints[offset + i];
                        joints16[i] = {uint16_t(j.x), uint16_t(j.y), uint16_t(j.z), uint16_t(j.w)};
                    }
                    const uint32_t jacc = AddDataAccessor(std::span<const std::array<uint16_t, 4>>(joints16), fastgltf::AccessorType::Vec4, fastgltf::ComponentType::UnsignedShort, fastgltf::BufferTarget::ArrayBuffer);
                    prim_attrs.emplace_back(fastgltf::Attribute{"JOINTS_0", jacc});

                    emit_vec4("WEIGHTS_0", mesh.DeformData->Weights.data() + offset);
                }

                std::pmr::vector<fastgltf::pmr::SmallVector<fastgltf::Attribute, 4>> prim_targets;
                if (target_count > 0) {
                    prim_targets.reserve(target_count);
                    for (uint32_t t = 0; t < target_count; ++t) {
                        fastgltf::pmr::SmallVector<fastgltf::Attribute, 4> tattrs;
                        const std::span<const vec3> pos_deltas(mesh.MorphData->PositionDeltas.data() + std::size_t(t) * total_vcount + offset, pcount);
                        tattrs.emplace_back(fastgltf::Attribute{"POSITION", AddVec3Accessor(pos_deltas, false, fastgltf::BufferTarget::ArrayBuffer)});
                        if (has_normal_deltas) {
                            const std::span<const vec3> norm_deltas(mesh.MorphData->NormalDeltas.data() + std::size_t(t) * total_vcount + offset, pcount);
                            tattrs.emplace_back(fastgltf::Attribute{"NORMAL", AddVec3Accessor(norm_deltas, false, fastgltf::BufferTarget::ArrayBuffer)});
                        }
                        if (has_tangent_deltas) {
                            const std::span<const vec3> tan_deltas(mesh.MorphData->TangentDeltas.data() + std::size_t(t) * total_vcount + offset, pcount);
                            tattrs.emplace_back(fastgltf::Attribute{"TANGENT", AddVec3Accessor(tan_deltas, false, fastgltf::BufferTarget::ArrayBuffer)});
                        }
                        prim_targets.emplace_back(std::move(tattrs));
                    }
                }

                // Face indices are absolute into the shared buffer; rebase into this primitive's slice.
                const auto face_it = faces_by_prim.find(prim_idx);
                std::vector<uint32_t> indices;
                if (face_it != faces_by_prim.end()) {
                    indices.reserve(face_it->second.size() * 3);
                    for (const uint32_t fi : face_it->second) {
                        const auto &face = tri.Faces[fi];
                        if (face.size() < 3) continue;
                        // Fan-triangulate polygons (source is usually already triangulated).
                        for (std::size_t k = 1; k + 1 < face.size(); ++k) {
                            indices.emplace_back(face[0] - offset);
                            indices.emplace_back(face[k] - offset);
                            indices.emplace_back(face[k + 1] - offset);
                        }
                    }
                }
                if (indices.empty()) continue;

                // Empty HasSourceIndices falls back to emitting (preserves legacy behavior).
                const bool emit_indices = prim_idx < tprims.HasSourceIndices.size() ? tprims.HasSourceIndices[prim_idx] != 0 : true;
                fastgltf::Optional<std::size_t> indices_accessor;
                if (emit_indices) {
                    indices_accessor = AddDataAccessor(std::span<const uint32_t>(indices), fastgltf::AccessorType::Scalar, fastgltf::ComponentType::UnsignedInt, fastgltf::BufferTarget::ElementArrayBuffer);
                }

                fastgltf::Optional<std::size_t> material_index;
                if (prim_idx < tprims.MaterialIndices.size()) {
                    const uint32_t mat = tprims.MaterialIndices[prim_idx];
                    if (mat < material_count) material_index = mat;
                }

                std::vector<fastgltf::Optional<std::size_t>> mappings;
                if (prim_idx < tprims.VariantMappings.size()) {
                    for (const auto &m : tprims.VariantMappings[prim_idx]) {
                        if (m.has_value() && *m < material_count) mappings.emplace_back(std::size_t(*m));
                        else mappings.emplace_back();
                    }
                }

                primitives.emplace_back(fastgltf::Primitive{
                    .attributes = std::move(prim_attrs),
                    .type = fastgltf::PrimitiveType::Triangles,
                    .targets = std::move(prim_targets),
                    .indicesAccessor = indices_accessor,
                    .materialIndex = material_index,
                    .mappings = std::move(mappings),
                    .dracoCompression = nullptr,
                });
            }
        }

        const auto emit_non_triangle_attrs = [&](fastgltf::pmr::SmallVector<fastgltf::Attribute, 4> &out, std::size_t vcount, const ::MeshVertexAttributes &va) {
            if (va.Normals && va.Normals->size() == vcount) {
                out.emplace_back(fastgltf::Attribute{"NORMAL", AddVec3Accessor(*va.Normals, false, fastgltf::BufferTarget::ArrayBuffer)});
            }
            if (va.Colors0 && va.Colors0->size() == vcount) {
                EmitColor0(out, std::span<const vec4>(va.Colors0->data(), vcount), va.Colors0ComponentCount);
            }
        };

        // --- Line primitive ---
        if (mesh.Lines && !mesh.Lines->Positions.empty() && !mesh.Lines->Edges.empty()) {
            const auto &l = *mesh.Lines;
            const uint32_t pos_acc = AddVec3Accessor(l.Positions, true, fastgltf::BufferTarget::ArrayBuffer);

            std::vector<uint32_t> idx;
            idx.reserve(l.Edges.size() * 2);
            for (const auto &pair : l.Edges) {
                idx.emplace_back(pair[0]);
                idx.emplace_back(pair[1]);
            }
            const uint32_t iacc = AddDataAccessor(std::span<const uint32_t>(idx), fastgltf::AccessorType::Scalar, fastgltf::ComponentType::UnsignedInt, fastgltf::BufferTarget::ElementArrayBuffer);

            fastgltf::pmr::SmallVector<fastgltf::Attribute, 4> attrs;
            attrs.emplace_back(fastgltf::Attribute{"POSITION", pos_acc});
            emit_non_triangle_attrs(attrs, l.Positions.size(), mesh.LineAttrs);
            primitives.emplace_back(fastgltf::Primitive{
                .attributes = std::move(attrs),
                .type = fastgltf::PrimitiveType::Lines,
                .targets = {},
                .indicesAccessor = iacc,
                .materialIndex = {},
                .mappings = {},
                .dracoCompression = nullptr,
            });
        }

        // --- Point primitive ---
        if (mesh.Points && !mesh.Points->Positions.empty()) {
            const auto &p = *mesh.Points;
            const uint32_t pos_acc = AddVec3Accessor(p.Positions, true, fastgltf::BufferTarget::ArrayBuffer);
            fastgltf::pmr::SmallVector<fastgltf::Attribute, 4> attrs;
            attrs.emplace_back(fastgltf::Attribute{"POSITION", pos_acc});
            emit_non_triangle_attrs(attrs, p.Positions.size(), mesh.PointAttrs);
            primitives.emplace_back(fastgltf::Primitive{
                .attributes = std::move(attrs),
                .type = fastgltf::PrimitiveType::Points,
                .targets = {},
                .indicesAccessor = {},
                .materialIndex = {},
                .mappings = {},
                .dracoCompression = nullptr,
            });
        }

        // glTF requires >= 1 primitive per mesh; emit a single degenerate point.
        if (primitives.empty()) {
            const vec3 stub{0.f, 0.f, 0.f};
            const uint32_t pos_acc = AddVec3Accessor(std::span<const vec3>(&stub, 1), true, fastgltf::BufferTarget::ArrayBuffer);
            fastgltf::pmr::SmallVector<fastgltf::Attribute, 4> attrs;
            attrs.emplace_back(fastgltf::Attribute{"POSITION", pos_acc});
            primitives.emplace_back(fastgltf::Primitive{
                .attributes = std::move(attrs),
                .type = fastgltf::PrimitiveType::Points,
                .targets = {},
                .indicesAccessor = {},
                .materialIndex = {},
                .mappings = {},
                .dracoCompression = nullptr,
            });
        }

        fastgltf::pmr::MaybeSmallVector<fastgltf::num> weights;
        if (mesh.MorphData && !mesh.MorphData->DefaultWeights.empty()) {
            weights.reserve(mesh.MorphData->DefaultWeights.size());
            for (const float w : mesh.MorphData->DefaultWeights) weights.emplace_back(w);
        }

        asset.meshes.emplace_back(fastgltf::Mesh{
            .primitives = std::move(primitives),
            .weights = std::move(weights),
            .name = ToFgStr(mesh.Name),
        });
    }

    // Source→dense skin index remap for node refs below.
    std::unordered_map<uint32_t, uint32_t> skin_remap;
    for (uint32_t j = 0; j < scene.Skins.size(); ++j) skin_remap[scene.Skins[j].SkinIndex] = j;
    asset.skins.reserve(scene.Skins.size());
    for (const auto &s : scene.Skins) {
        fastgltf::pmr::MaybeSmallVector<std::size_t> joints;
        joints.reserve(s.Joints.size());
        for (const auto &joint : s.Joints) joints.emplace_back(joint.JointNodeIndex);

        const auto ibm = [&]() -> fastgltf::Optional<std::size_t> {
            if (s.InverseBindMatrices.empty()) return {};
            return AddDataAccessor(std::span<const mat4>(s.InverseBindMatrices), fastgltf::AccessorType::Mat4, fastgltf::ComponentType::Float);
        }();

        asset.skins.emplace_back(fastgltf::Skin{
            .inverseBindMatrices = ibm,
            .skeleton = ToFgOpt<std::size_t>(s.SkeletonNodeIndex),
            .joints = std::move(joints),
            .name = ToFgStr(s.Name),
        });
    }

    // Animations
    asset.animations.reserve(scene.Animations.size());
    for (const auto &clip : scene.Animations) {
        fastgltf::pmr::MaybeSmallVector<fastgltf::AnimationSampler> samplers;
        fastgltf::pmr::MaybeSmallVector<fastgltf::AnimationChannel> channels;
        for (const auto &ch : clip.Channels) {
            if (ch.TimesSeconds.empty()) continue;
            // Input (times): spec requires min/max on the accessor.
            const uint32_t t_offset = AppendAligned<float>(bin, ch.TimesSeconds);
            const uint32_t t_bv = AddBufferView(t_offset, uint32_t(ch.TimesSeconds.size() * sizeof(float)));
            const auto [t_min, t_max] = std::minmax_element(ch.TimesSeconds.begin(), ch.TimesSeconds.end());
            const uint32_t t_acc = AddAccessor(
                t_bv, uint32_t(ch.TimesSeconds.size()), fastgltf::AccessorType::Scalar, fastgltf::ComponentType::Float,
                MakeBounds({double(*t_min)}), MakeBounds({double(*t_max)})
            );

            // Output (values). CubicSpline triples are already flattened into Values.
            const uint32_t v_offset = AppendAligned<float>(bin, ch.Values);
            const uint32_t v_bv = AddBufferView(v_offset, uint32_t(ch.Values.size() * sizeof(float)));
            const auto [v_type, v_count] = [&] -> std::pair<fastgltf::AccessorType, uint32_t> {
                switch (ch.Target) {
                    case AnimationPath::Translation:
                    case AnimationPath::Scale: return {fastgltf::AccessorType::Vec3, uint32_t(ch.Values.size() / 3)};
                    case AnimationPath::Rotation: return {fastgltf::AccessorType::Vec4, uint32_t(ch.Values.size() / 4)};
                    case AnimationPath::Weights: return {fastgltf::AccessorType::Scalar, uint32_t(ch.Values.size())};
                }
            }();
            const uint32_t v_acc = AddAccessor(v_bv, v_count, v_type, fastgltf::ComponentType::Float);

            samplers.emplace_back(fastgltf::AnimationSampler{
                .inputAccessor = t_acc,
                .outputAccessor = v_acc,
                .interpolation = FromInterp(ch.Interp),
            });
            channels.emplace_back(fastgltf::AnimationChannel{
                .samplerIndex = samplers.size() - 1,
                .nodeIndex = ch.TargetNodeIndex,
                .path = FromPath(ch.Target),
            });
        }
        asset.animations.emplace_back(fastgltf::Animation{
            .channels = std::move(channels),
            .samplers = std::move(samplers),
            .name = ToFgStr(clip.Name),
        });
    }

    // Cameras
    asset.cameras.reserve(scene.Cameras.size());
    for (const auto &c : scene.Cameras) {
        auto camera = std::visit(
            [](const auto &proj) -> std::variant<fastgltf::Camera::Perspective, fastgltf::Camera::Orthographic> {
                using P = std::decay_t<decltype(proj)>;
                if constexpr (std::is_same_v<P, Perspective>) {
                    return fastgltf::Camera::Perspective{
                        .aspectRatio = ToFgOpt<fastgltf::num>(proj.AspectRatio),
                        .yfov = proj.FieldOfViewRad,
                        .zfar = ToFgOpt<fastgltf::num>(proj.FarClip),
                        .znear = proj.NearClip,
                    };
                } else {
                    return fastgltf::Camera::Orthographic{
                        .xmag = proj.Mag.x,
                        .ymag = proj.Mag.y,
                        .zfar = proj.FarClip,
                        .znear = proj.NearClip,
                    };
                }
            },
            c.Camera
        );
        asset.cameras.emplace_back(fastgltf::Camera{
            .camera = std::move(camera),
            .name = ToFgStr(c.Name),
        });
    }

    // Lights (KHR_lights_punctual)
    asset.lights.reserve(scene.Lights.size());
    for (const auto &l : scene.Lights) {
        const auto type = l.Light.Type == PunctualLightType::Point ? fastgltf::LightType::Point : l.Light.Type == PunctualLightType::Spot ? fastgltf::LightType::Spot :
                                                                                                                                            fastgltf::LightType::Directional;
        const bool is_spot = type == fastgltf::LightType::Spot;
        asset.lights.emplace_back(fastgltf::Light{
            .type = type,
            .color = {l.Light.Color.x, l.Light.Color.y, l.Light.Color.z},
            .intensity = l.Light.Intensity,
            .range = (type != fastgltf::LightType::Directional && l.Light.Range > 0) ? fastgltf::Optional<fastgltf::num>{l.Light.Range} : fastgltf::Optional<fastgltf::num>{},
            .innerConeAngle = is_spot ? fastgltf::Optional<fastgltf::num>{std::acos(std::clamp(l.Light.InnerConeCos, -1.f, 1.f))} : fastgltf::Optional<fastgltf::num>{},
            .outerConeAngle = is_spot ? fastgltf::Optional<fastgltf::num>{std::acos(std::clamp(l.Light.OuterConeCos, -1.f, 1.f))} : fastgltf::Optional<fastgltf::num>{},
            .name = ToFgStr(l.Name),
        });
    }

    // Nodes referenced by multiple Objects become EXT_mesh_gpu_instancing.
    std::vector<std::vector<uint32_t>> node_objects(scene.Nodes.size());
    for (uint32_t oi = 0; oi < scene.Objects.size(); ++oi) {
        const auto ni = scene.Objects[oi].NodeIndex;
        if (ni < node_objects.size()) node_objects[ni].emplace_back(oi);
    }

    // KHR_implicit_shapes: dedupe primitive shapes referenced by colliders/triggers into asset.shapes.
    using ShapeKey = std::tuple<uint8_t, float, float, float, float>;
    const auto to_fg_shape = [](const PhysicsShape &s) -> std::optional<fastgltf::Shape> {
        return std::visit(
            overloaded{
                [](const physics::Box &b) -> std::optional<fastgltf::Shape> {
                    return fastgltf::BoxShape{.size = {b.Size.x, b.Size.y, b.Size.z}};
                },
                [](const physics::Sphere &s) -> std::optional<fastgltf::Shape> {
                    return fastgltf::SphereShape{.radius = s.Radius};
                },
                [](const physics::Capsule &c) -> std::optional<fastgltf::Shape> {
                    return fastgltf::CapsuleShape{.height = c.Height, .radiusBottom = c.RadiusBottom, .radiusTop = c.RadiusTop};
                },
                [](const physics::Cylinder &c) -> std::optional<fastgltf::Shape> {
                    return fastgltf::CylinderShape{.height = c.Height, .radiusBottom = c.RadiusBottom, .radiusTop = c.RadiusTop};
                },
                [](const physics::Plane &p) -> std::optional<fastgltf::Shape> {
                    return fastgltf::PlaneShape{.sizeX = p.SizeX, .sizeZ = p.SizeZ, .doubleSided = p.DoubleSided};
                },
                [](const auto &) -> std::optional<fastgltf::Shape> { return std::nullopt; },
            },
            s
        );
    };
    const auto shape_key = [](const fastgltf::Shape &s) -> ShapeKey {
        return std::visit(
            overloaded{
                [](const fastgltf::BoxShape &b) { return ShapeKey{0, b.size[0], b.size[1], b.size[2], 0}; },
                [](const fastgltf::SphereShape &s) { return ShapeKey{1, float(s.radius), 0, 0, 0}; },
                [](const fastgltf::CapsuleShape &c) { return ShapeKey{2, float(c.height), float(c.radiusBottom), float(c.radiusTop), 0}; },
                [](const fastgltf::CylinderShape &c) { return ShapeKey{3, float(c.height), float(c.radiusBottom), float(c.radiusTop), 0}; },
                [](const fastgltf::PlaneShape &p) { return ShapeKey{4, float(p.sizeX), float(p.sizeZ), p.doubleSided ? 1.f : 0.f, 0}; },
            },
            s
        );
    };
    std::map<ShapeKey, std::size_t> shape_index_by_key;
    const auto emit_shape_index = [&](const fastgltf::Shape &s) -> std::size_t {
        const auto key = shape_key(s);
        if (auto it = shape_index_by_key.find(key); it != shape_index_by_key.end()) return it->second;
        asset.shapes.emplace_back(s);
        const auto idx = asset.shapes.size() - 1;
        shape_index_by_key[key] = idx;
        return idx;
    };

    // Nodes
    asset.nodes.reserve(scene.Nodes.size());
    bool uses_gpu_instancing = false;
    bool uses_physics_rigid_bodies = false;
    for (uint32_t ni = 0; ni < scene.Nodes.size(); ++ni) {
        const auto &n = scene.Nodes[ni];

        const auto skin_index = [&]() -> fastgltf::Optional<std::size_t> {
            if (!n.SkinIndex) return {};
            const auto it = skin_remap.find(*n.SkinIndex);
            if (it == skin_remap.end()) return {};
            return it->second;
        }();

        fastgltf::pmr::MaybeSmallVector<std::size_t> children(n.ChildrenNodeIndices.begin(), n.ChildrenNodeIndices.end());

        // EXT_mesh_gpu_instancing. Per-instance TRS = node.WorldTransform^-1 * object.WorldTransform.
        // Emit only channels that aren't uniformly default; spec requires >=1 attribute so fall back
        // to TRANSLATION when everything's default.
        const bool needs_instancing = n.InScene && n.MeshIndex && node_objects[ni].size() > 1;
        if (needs_instancing) uses_gpu_instancing = true;
        std::pmr::vector<fastgltf::Attribute> instancing;
        if (needs_instancing) {
            const auto &obj_indices = node_objects[ni];
            const uint32_t count = uint32_t(obj_indices.size());
            const mat4 node_world_inv = glm::inverse(ToMatrix(n.WorldTransform));

            std::vector<vec3> translations(count);
            std::vector<vec4> rotations(count); // xyzw
            std::vector<vec3> scales(count);
            bool any_t = false, any_r = false, any_s = false;
            for (uint32_t i = 0; i < count; ++i) {
                const auto &obj = scene.Objects[obj_indices[i]];
                const Transform local = ToTransform(node_world_inv * ToMatrix(obj.WorldTransform));
                translations[i] = local.P;
                rotations[i] = {local.R.x, local.R.y, local.R.z, local.R.w};
                scales[i] = local.S;
                if (local.P != vec3{0.f}) any_t = true;
                if (local.R != quat{1, 0, 0, 0}) any_r = true;
                if (local.S != vec3{1.f}) any_s = true;
            }

            const auto add_vec3 = [&](const char *name, std::span<const vec3> data) {
                instancing.emplace_back(fastgltf::Attribute{name, AddDataAccessor(data, fastgltf::AccessorType::Vec3, fastgltf::ComponentType::Float)});
            };
            if (any_t) add_vec3("TRANSLATION", translations);
            if (any_r) {
                const uint32_t acc = AddDataAccessor(std::span<const vec4>(rotations), fastgltf::AccessorType::Vec4, fastgltf::ComponentType::Float);
                instancing.emplace_back(fastgltf::Attribute{"ROTATION", acc});
            }
            if (any_s) add_vec3("SCALE", scales);
            if (!any_t && !any_r && !any_s) add_vec3("TRANSLATION", translations);
        }

        std::unique_ptr<fastgltf::PhysicsRigidBody> physics_rigid_body;
        const bool has_physics = n.Motion || n.Collider || n.Trigger || n.Joint || n.ColliderGeometryMeshIndex;
        if (has_physics) {
            physics_rigid_body = std::make_unique<fastgltf::PhysicsRigidBody>();
            uses_physics_rigid_bodies = true;

            if (n.Motion) {
                fastgltf::Motion motion{};
                motion.isKinematic = n.Motion->IsKinematic;
                if (n.Motion->Mass) motion.mass = fastgltf::Optional<fastgltf::num>{*n.Motion->Mass};
                if (n.Motion->CenterOfMass) motion.centerOfMass = {n.Motion->CenterOfMass->x, n.Motion->CenterOfMass->y, n.Motion->CenterOfMass->z};
                if (n.Motion->InertiaDiagonal) {
                    motion.inertialDiagonal = fastgltf::Optional<fastgltf::math::fvec3>{fastgltf::math::fvec3{n.Motion->InertiaDiagonal->x, n.Motion->InertiaDiagonal->y, n.Motion->InertiaDiagonal->z}};
                }
                if (n.Motion->InertiaOrientation) {
                    motion.inertialOrientation = fastgltf::Optional<fastgltf::math::fvec4>{fastgltf::math::fvec4{n.Motion->InertiaOrientation->x, n.Motion->InertiaOrientation->y, n.Motion->InertiaOrientation->z, n.Motion->InertiaOrientation->w}};
                }
                motion.gravityFactor = n.Motion->GravityFactor;
                if (n.Velocity) {
                    motion.linearVelocity = {n.Velocity->Linear.x, n.Velocity->Linear.y, n.Velocity->Linear.z};
                    motion.angularVelocity = {n.Velocity->Angular.x, n.Velocity->Angular.y, n.Velocity->Angular.z};
                }
                physics_rigid_body->motion = fastgltf::Optional<fastgltf::Motion>{std::move(motion)};
            }

            if (n.Collider || n.ColliderGeometryMeshIndex) {
                fastgltf::Collider collider{};
                if (n.Collider) {
                    if (auto fg_shape = to_fg_shape(n.Collider->Shape)) {
                        collider.geometry.shape = emit_shape_index(*fg_shape);
                    } else if (std::holds_alternative<physics::ConvexHull>(n.Collider->Shape)) {
                        if (n.ColliderGeometryMeshIndex) collider.geometry.mesh = *n.ColliderGeometryMeshIndex;
                        collider.geometry.convexHull = true;
                    } else if (std::holds_alternative<physics::TriangleMesh>(n.Collider->Shape)) {
                        if (n.ColliderGeometryMeshIndex) collider.geometry.mesh = *n.ColliderGeometryMeshIndex;
                    }
                } else {
                    collider.geometry.mesh = *n.ColliderGeometryMeshIndex;
                }
                if (n.Material) {
                    if (n.Material->PhysicsMaterialIndex) collider.physicsMaterial = *n.Material->PhysicsMaterialIndex;
                    if (n.Material->CollisionFilterIndex) collider.collisionFilter = *n.Material->CollisionFilterIndex;
                }
                physics_rigid_body->collider = fastgltf::Optional<fastgltf::Collider>{std::move(collider)};
            }

            if (n.Trigger) {
                using TriggerVariant = std::variant<fastgltf::GeometryTrigger, fastgltf::NodeTrigger>;
                if (n.Trigger->Shape) {
                    fastgltf::GeometryTrigger gt{};
                    if (auto fg_shape = to_fg_shape(*n.Trigger->Shape)) {
                        gt.geometry.shape = emit_shape_index(*fg_shape);
                    } else if (n.Trigger->GeometryMeshIndex) {
                        gt.geometry.mesh = *n.Trigger->GeometryMeshIndex;
                        gt.geometry.convexHull = std::holds_alternative<physics::ConvexHull>(*n.Trigger->Shape);
                    }
                    if (n.Trigger->CollisionFilterIndex) gt.collisionFilter = *n.Trigger->CollisionFilterIndex;
                    physics_rigid_body->trigger = fastgltf::Optional<TriggerVariant>{TriggerVariant{std::move(gt)}};
                } else if (!n.Trigger->NodeIndices.empty()) {
                    fastgltf::NodeTrigger nt;
                    for (const auto idx : n.Trigger->NodeIndices) nt.nodes.emplace_back(idx);
                    physics_rigid_body->trigger = fastgltf::Optional<TriggerVariant>{TriggerVariant{std::move(nt)}};
                }
            }

            if (n.Joint) {
                fastgltf::Joint joint{};
                joint.connectedNode = n.Joint->ConnectedNodeIndex;
                joint.joint = n.Joint->JointDefIndex;
                joint.enableCollision = n.Joint->EnableCollision;
                physics_rigid_body->joint = fastgltf::Optional<fastgltf::Joint>{std::move(joint)};
            }
        }

        // Matrix-form source round-trips as matrix; TRS-form emits from LocalTransform.
        const auto fg_transform = [&]() -> std::variant<fastgltf::TRS, fastgltf::math::fmat4x4> {
            if (n.SourceMatrix) {
                fastgltf::math::fmat4x4 out;
                for (std::size_t c = 0; c < 4; ++c) {
                    for (std::size_t r2 = 0; r2 < 4; ++r2) out[c][r2] = (*n.SourceMatrix)[c][r2];
                }
                return out;
            }
            return fastgltf::TRS{
                .translation = {n.LocalTransform.P.x, n.LocalTransform.P.y, n.LocalTransform.P.z},
                .rotation = fastgltf::math::fquat(n.LocalTransform.R.x, n.LocalTransform.R.y, n.LocalTransform.R.z, n.LocalTransform.R.w),
                .scale = {n.LocalTransform.S.x, n.LocalTransform.S.y, n.LocalTransform.S.z},
            };
        }();

        asset.nodes.emplace_back(fastgltf::Node{
            .meshIndex = ToFgOpt<std::size_t>(n.MeshIndex),
            .skinIndex = skin_index,
            .cameraIndex = ToFgOpt<std::size_t>(n.CameraIndex),
            .lightIndex = ToFgOpt<std::size_t>(n.LightIndex),
            .children = std::move(children),
            .weights = {},
            .transform = fg_transform,
            .instancingAttributes = std::move(instancing),
            .name = ToFgStr(n.Name),
            .physicsRigidBody = std::move(physics_rigid_body),
            .visible = true,
            .selectable = true,
            .hoverable = true,
        });
    }

    fastgltf::pmr::MaybeSmallVector<std::size_t> scene_roots;
    if (!scene.DefaultSceneRoots.empty()) {
        scene_roots.reserve(scene.DefaultSceneRoots.size());
        for (const auto n : scene.DefaultSceneRoots) scene_roots.emplace_back(n);
    } else {
        for (uint32_t ni = 0; ni < scene.Nodes.size(); ++ni) {
            const auto &n = scene.Nodes[ni];
            if (!n.InScene) continue;
            if (n.ParentNodeIndex && *n.ParentNodeIndex < scene.Nodes.size() && scene.Nodes[*n.ParentNodeIndex].InScene) continue;
            scene_roots.emplace_back(ni);
        }
    }
    // EXT_lights_image_based
    fastgltf::Optional<std::size_t> scene_ibl_index;
    if (scene.ImageBasedLight) {
        const auto &src_ibl = *scene.ImageBasedLight;
        fastgltf::ImageBasedLight ibl{
            .intensity = src_ibl.Intensity,
            .rotation = fastgltf::math::fquat(src_ibl.Rotation.x, src_ibl.Rotation.y, src_ibl.Rotation.z, src_ibl.Rotation.w),
            .specularImageSize = src_ibl.SpecularImageSize,
            .specularImages = {},
            .irradianceCoefficients = {},
            .name = ToFgStr(src_ibl.Name),
        };
        ibl.specularImages.reserve(src_ibl.SpecularImageIndicesByMip.size());
        for (const auto &mip : src_ibl.SpecularImageIndicesByMip) {
            std::array<std::size_t, 6> faces{};
            for (std::size_t face = 0; face < 6; ++face) faces[face] = mip[face];
            ibl.specularImages.emplace_back(faces);
        }
        if (src_ibl.IrradianceCoefficients) {
            std::array<fastgltf::math::fvec3, 9> coeffs{};
            for (std::size_t i = 0; i < 9; ++i) {
                const auto &c = (*src_ibl.IrradianceCoefficients)[i];
                coeffs[i] = {c.x, c.y, c.z};
            }
            ibl.irradianceCoefficients = coeffs;
        }
        asset.imageBasedLights.emplace_back(std::move(ibl));
        scene_ibl_index = std::size_t{0};
    }

    asset.scenes.emplace_back(fastgltf::Scene{
        .nodeIndices = std::move(scene_roots),
        .imageBasedLightIndex = scene_ibl_index,
        .name = ToFgStr(scene.DefaultSceneName),
    });
    asset.defaultScene = 0;

    // Physics doc-level arrays
    asset.physicsMaterials.reserve(scene.PhysicsMaterials.size());
    for (const auto &pm : scene.PhysicsMaterials) {
        asset.physicsMaterials.emplace_back(fastgltf::PhysicsMaterial{
            .staticFriction = pm.StaticFriction,
            .dynamicFriction = pm.DynamicFriction,
            .restitution = pm.Restitution,
            .frictionCombine = FromCombine(pm.FrictionCombine),
            .restitutionCombine = FromCombine(pm.RestitutionCombine),
        });
    }
    asset.collisionFilters.reserve(scene.CollisionFilters.size());
    for (const auto &cf : scene.CollisionFilters) {
        const auto to_fg_strs = [](const std::vector<std::string> &xs) {
            fastgltf::pmr::MaybeSmallVector<FgString> out;
            out.reserve(xs.size());
            for (const auto &s : xs) out.emplace_back(ToFgStr(s));
            return out;
        };
        asset.collisionFilters.emplace_back(fastgltf::CollisionFilter{
            .collisionSystems = to_fg_strs(cf.CollisionSystems),
            .notCollideWithSystems = to_fg_strs(cf.NotCollideWithSystems),
            .collideWithSystems = to_fg_strs(cf.CollideWithSystems),
        });
    }
    asset.physicsJoints.reserve(scene.PhysicsJointDefs.size());
    for (const auto &jd : scene.PhysicsJointDefs) {
        fastgltf::pmr::MaybeSmallVector<fastgltf::JointLimit> limits;
        limits.reserve(jd.Limits.size());
        for (const auto &lim : jd.Limits) {
            fastgltf::pmr::SmallVector<uint8_t, 3> linear_axes;
            for (const auto a : lim.LinearAxes) linear_axes.emplace_back(a);
            fastgltf::pmr::SmallVector<uint8_t, 3> angular_axes;
            for (const auto a : lim.AngularAxes) angular_axes.emplace_back(a);
            limits.emplace_back(fastgltf::JointLimit{
                .linearAxes = std::move(linear_axes),
                .angularAxes = std::move(angular_axes),
                .min = ToFgOpt<fastgltf::num>(lim.Min),
                .max = ToFgOpt<fastgltf::num>(lim.Max),
                .stiffness = ToFgOpt<fastgltf::num>(lim.Stiffness),
                .damping = lim.Damping,
            });
        }
        fastgltf::pmr::MaybeSmallVector<fastgltf::JointDrive> drives;
        drives.reserve(jd.Drives.size());
        for (const auto &drv : jd.Drives) {
            drives.emplace_back(fastgltf::JointDrive{
                .type = drv.Type == PhysicsDriveType::Angular ? fastgltf::DriveType::Angular : fastgltf::DriveType::Linear,
                .mode = drv.Mode == PhysicsDriveMode::Acceleration ? fastgltf::DriveMode::Acceleration : fastgltf::DriveMode::Force,
                .axis = drv.Axis,
                .maxForce = drv.MaxForce,
                .positionTarget = drv.PositionTarget,
                .velocityTarget = drv.VelocityTarget,
                .stiffness = drv.Stiffness,
                .damping = drv.Damping,
            });
        }
        asset.physicsJoints.emplace_back(fastgltf::PhysicsJoint{
            .limits = std::move(limits),
            .drives = std::move(drives),
        });
    }

    asset.extensionsRequired.reserve(scene.ExtensionsRequired.size());
    for (const auto &e : scene.ExtensionsRequired) asset.extensionsRequired.emplace_back(e);

    // extensionsUsed
    if (!asset.lights.empty()) asset.extensionsUsed.emplace_back("KHR_lights_punctual");
    if (uses_gpu_instancing) asset.extensionsUsed.emplace_back("EXT_mesh_gpu_instancing");
    if (uses_physics_rigid_bodies || !asset.physicsMaterials.empty() || !asset.collisionFilters.empty() || !asset.physicsJoints.empty()) {
        asset.extensionsUsed.emplace_back("KHR_physics_rigid_bodies");
    }
    if (!asset.shapes.empty()) asset.extensionsUsed.emplace_back("KHR_implicit_shapes");
    if (!asset.imageBasedLights.empty()) asset.extensionsUsed.emplace_back("EXT_lights_image_based");
    const auto any_material = [&](auto pred) { return std::ranges::any_of(asset.materials, pred); };
    if (any_material([](const auto &m) { return m.unlit; })) asset.extensionsUsed.emplace_back("KHR_materials_unlit");
    if (any_material([](const auto &m) { return m.ior.has_value(); })) asset.extensionsUsed.emplace_back("KHR_materials_ior");
    if (any_material([](const auto &m) { return m.emissiveStrength.has_value(); })) asset.extensionsUsed.emplace_back("KHR_materials_emissive_strength");
    if (any_material([](const auto &m) { return m.dispersion.has_value(); })) asset.extensionsUsed.emplace_back("KHR_materials_dispersion");
    if (any_material([](const auto &m) { return m.sheen != nullptr; })) asset.extensionsUsed.emplace_back("KHR_materials_sheen");
    if (any_material([](const auto &m) { return m.specular != nullptr; })) asset.extensionsUsed.emplace_back("KHR_materials_specular");
    if (any_material([](const auto &m) { return m.transmission != nullptr; })) asset.extensionsUsed.emplace_back("KHR_materials_transmission");
    if (any_material([](const auto &m) { return m.diffuseTransmission != nullptr; })) asset.extensionsUsed.emplace_back("KHR_materials_diffuse_transmission");
    if (any_material([](const auto &m) { return m.volume != nullptr; })) asset.extensionsUsed.emplace_back("KHR_materials_volume");
    if (any_material([](const auto &m) { return m.clearcoat != nullptr; })) asset.extensionsUsed.emplace_back("KHR_materials_clearcoat");
    if (any_material([](const auto &m) { return m.anisotropy != nullptr; })) asset.extensionsUsed.emplace_back("KHR_materials_anisotropy");
    if (any_material([](const auto &m) { return m.iridescence != nullptr; })) asset.extensionsUsed.emplace_back("KHR_materials_iridescence");
    if (!scene.MaterialVariants.empty()) {
        asset.materialVariants.reserve(scene.MaterialVariants.size());
        for (const auto &v : scene.MaterialVariants) asset.materialVariants.emplace_back(v);
        asset.extensionsUsed.emplace_back("KHR_materials_variants");
    }
    // KHR_texture_transform: emitted per-texture when non-identity or when source had it at all.
    {
        const auto any_transform = [](const ::TextureInfo &ti, const TextureTransformMeta *meta = nullptr) {
            if (ti.Slot == InvalidSlot) return false;
            if (ti.UvOffset.x != 0.f || ti.UvOffset.y != 0.f || ti.UvScale.x != 1.f || ti.UvScale.y != 1.f || ti.UvRotation != 0.f) return true;
            return meta && meta->SourceHadExtension;
        };
        bool uses_texture_transform = false;
        for (uint32_t i = 0; i < material_count && !uses_texture_transform; ++i) {
            const auto &m = scene.Materials[i].Value;
            const auto any_of = [&](std::initializer_list<const ::TextureInfo *> texs) {
                for (const auto *t : texs)
                    if (any_transform(*t)) return true;
                return false;
            };
            if (any_transform(m.BaseColorTexture, &m.BaseColorMeta) || any_transform(m.MetallicRoughnessTexture, &m.MetallicRoughnessMeta) ||
                any_transform(m.NormalTexture, &m.NormalMeta) || any_transform(m.OcclusionTexture, &m.OcclusionMeta) || any_transform(m.EmissiveTexture, &m.EmissiveMeta) ||
                (m.Sheen && any_of({&m.Sheen->ColorTexture, &m.Sheen->RoughnessTexture})) ||
                (m.Specular && any_of({&m.Specular->Texture, &m.Specular->ColorTexture})) ||
                (m.Transmission && any_transform(m.Transmission->Texture)) ||
                (m.DiffuseTransmission && any_of({&m.DiffuseTransmission->Texture, &m.DiffuseTransmission->ColorTexture})) ||
                (m.Volume && any_transform(m.Volume->ThicknessTexture)) ||
                (m.Clearcoat && any_of({&m.Clearcoat->Texture, &m.Clearcoat->RoughnessTexture, &m.Clearcoat->NormalTexture})) ||
                (m.Anisotropy && any_transform(m.Anisotropy->Texture)) ||
                (m.Iridescence && any_of({&m.Iridescence->Texture, &m.Iridescence->ThicknessTexture}))) {
                uses_texture_transform = true;
            }
        }
        if (uses_texture_transform) asset.extensionsUsed.emplace_back("KHR_texture_transform");
    }

    // Finalize buffer. sources::Vector owns our binary blob; FileExporter writes it as a sibling .bin.
    asset.buffers.emplace_back(fastgltf::Buffer{
        .byteLength = bin.size(),
        .data = fastgltf::sources::Vector{.bytes = std::move(bin), .mimeType = fastgltf::MimeType::None},
        .name = ToFgStr(path.stem().string()),
    });

    asset.accessors = std::move(accessors);
    asset.bufferViews = std::move(bufferViews);

    fastgltf::FileExporter exporter;
    if (!scene.ExtrasByEntity.empty()) {
        exporter.setUserPointer(const_cast<ExtrasMap *>(&scene.ExtrasByEntity));
        exporter.setExtrasWriteCallback(EmitExtras);
    }
    const auto ext = path.extension();
    const auto err = ext == ".glb" ? exporter.writeGltfBinary(asset, path) : exporter.writeGltfJson(asset, path);
    if (err != fastgltf::Error::None) {
        return std::unexpected{std::format("fastgltf export to '{}' failed: {}", path.string(), fastgltf::getErrorMessage(err))};
    }
    return {};
}
} // namespace gltf
