#include "GltfScene.h"
#include "GltfConvert.h"

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
#include "mesh/MeshBatch.h"
#include "mesh/MeshComponents.h"
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
#include <entt/entity/registry.hpp>
#include <fastgltf/core.hpp>
#include <simdjson.h>

#include <bit>
#include <numbers>
#include <numeric>
#include <unordered_set>

namespace gltf {
using namespace detail;
namespace {
// Batches parsed geometry for one arena reservation before ECS insertion.
struct MeshData {
    std::optional<::MeshData> Triangles, Lines, Points;
    ::MeshVertexAttributes TriangleAttrs, LineAttrs, PointAttrs;
    ::MeshPrimitives TrianglePrimitives, LinePrimitives, PointPrimitives;
    std::optional<ArmatureDeformData> DeformData;
    std::optional<MorphTargetData> MorphData;
    std::string Name;
};

// Per-node KHR_physics_rigid_bodies staging data.
struct NodePhysics {
    std::optional<PhysicsMotion> Motion{};
    std::optional<PhysicsVelocity> Velocity{};
    std::optional<ColliderShape> Collider{};
    struct MaterialRefs {
        std::optional<uint32_t> PhysicsMaterialIndex{}, CollisionFilterIndex{};
    };
    std::optional<MaterialRefs> Material{};
    std::optional<uint32_t> ColliderGeometryMeshIndex{};

    struct TriggerData {
        std::optional<PhysicsShape> Shape{};
        std::optional<uint32_t> GeometryMeshIndex{};
        std::vector<uint32_t> NodeIndices{};
        std::optional<uint32_t> CollisionFilterIndex{};
    };
    std::optional<TriggerData> Trigger{};

    struct JointData {
        uint32_t ConnectedNodeIndex{};
        uint32_t JointDefIndex{};
        bool EnableCollision{false};
    };
    std::optional<JointData> Joint{};
};

struct Object {
    enum class Type : uint8_t {
        Empty,
        Mesh,
        Camera,
        Light,
    };

    Type ObjectType;
    uint32_t NodeIndex;
    std::optional<uint32_t> ParentNodeIndex;
    Transform LocalTransform;
    std::optional<uint32_t> MeshIndex, SkinIndex, CameraIndex, LightIndex;
    std::optional<std::vector<float>> NodeWeights;
    std::string Name;
};

void CollectExtras(simdjson::dom::object *extras, size_t idx, fastgltf::Category cat, void *userPtr) {
    if (!extras || !userPtr) return;
    static_cast<ExtrasMap *>(userPtr)->emplace(ExtrasKey(cat, idx), simdjson::minify(*extras));
}
std::optional<uint32_t> ToIndex(size_t index, size_t upper_bound) {
    if (index >= upper_bound) return {};
    return index;
}
std::optional<uint32_t> ToIndex(const fastgltf::Optional<size_t> &index, size_t upper_bound) {
    if (!index) return {};
    return ToIndex(*index, upper_bound);
}

Filter ToFilter(fastgltf::Filter f) { return MapEnum(FilterMap, f, Filter::LinearMipMapLinear); }
std::optional<Filter> ToFilter(const fastgltf::Optional<fastgltf::Filter> &filter) {
    if (!filter) return {};
    return ToFilter(*filter);
}
Wrap ToWrap(fastgltf::Wrap w) { return MapEnum(WrapMap, w, Wrap::Repeat); }
MimeType ToMimeType(fastgltf::MimeType m) { return MapEnum(MimeTypeMap, m, MimeType::None); }
AnimationInterpolation ToInterp(fastgltf::AnimationInterpolation i) { return MapEnum(InterpMap, i, AnimationInterpolation::Linear); }
PhysicsCombineMode ToCombineMode(fastgltf::CombineMode m) { return MapEnum(CombineMap, m, PhysicsCombineMode::Average); }

MaterialAlphaMode ToAlphaMode(fastgltf::AlphaMode m) { return MapEnum(AlphaModeMap, m, MaterialAlphaMode::Opaque); }

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

    Image image_result{.Bytes = {}, .MimeType = MimeType::None, .Name = std::string{image.name}};

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
                auto bytes = File::Read(image_path);
                if (!bytes) return std::unexpected{std::move(bytes.error())};
                // External images reload from SourceAbsPath after upload.
                image_result.Bytes = std::move(*bytes);
                image_result.MimeType = ToMimeType(uri.mimeType);
                image_result.SourceHadMimeType = uri.mimeType != fastgltf::MimeType::None;
                image_result.Uri = uri.uri.string();
                image_result.SourceAbsPath = image_path.string();
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
    if (image_result.MimeType == MimeType::None) image_result.MimeType = SniffMimeType(image_result.Bytes);
    return image_result;
}

// Appends a non-triangle primitive while preserving channel alignment across merged primitives.
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
    uint32_t &attribute_flags
) {
    attribute_flags = 0;
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

    // Packs morph deltas by primitive before conversion to per-target contiguous layout.
    if (!primitive.targets.empty()) {
        const uint32_t target_count = primitive.targets.size();
        const uint32_t prim_vertex_count = position_accessor.count;
        if (!morph) {
            morph.emplace();
            morph->TargetCount = target_count;
            morph->PositionDeltas.resize(target_count * base_vertex, vec3{0.f});
        }
        if (morph->TargetCount != target_count) return std::unexpected{"glTF primitive morph target count mismatch between primitives of the same mesh."};

        const auto prev_pos_size = morph->PositionDeltas.size();
        morph->PositionDeltas.resize(prev_pos_size + target_count * prim_vertex_count, vec3{0.f});
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
            morph->NormalDeltas.resize(prev_norm_size + target_count * prim_vertex_count, vec3{0.f});
        }
        if (prim_has_tangent_deltas && morph->TangentDeltas.empty() && prev_pos_size > 0) {
            morph->TangentDeltas.resize(prev_pos_size, vec3{0.f});
        }
        if (prim_has_tangent_deltas || !morph->TangentDeltas.empty()) {
            const auto prev_tan_size = morph->TangentDeltas.size();
            morph->TangentDeltas.resize(prev_tan_size + target_count * prim_vertex_count, vec3{0.f});
        }
        for (uint32_t t = 0; t < target_count; ++t) {
            if (const auto *pos_it = primitive.findTargetAttribute(t, "POSITION"); pos_it != primitive.targets[t].end()) {
                const auto &target_accessor = asset.accessors[pos_it->accessorIndex];
                if (target_accessor.count == prim_vertex_count) {
                    fastgltf::copyFromAccessor<vec3>(asset, target_accessor, &morph->PositionDeltas[prev_pos_size + t * prim_vertex_count]);
                }
            }
            if (!morph->NormalDeltas.empty()) {
                if (const auto *norm_it = primitive.findTargetAttribute(t, "NORMAL"); norm_it != primitive.targets[t].end()) {
                    const auto &norm_accessor = asset.accessors[norm_it->accessorIndex];
                    const auto prev_norm_size = morph->NormalDeltas.size() - target_count * prim_vertex_count;
                    if (norm_accessor.count == prim_vertex_count) {
                        fastgltf::copyFromAccessor<vec3>(asset, norm_accessor, &morph->NormalDeltas[prev_norm_size + t * prim_vertex_count]);
                    }
                }
            }
            if (!morph->TangentDeltas.empty()) {
                if (const auto *tan_it = primitive.findTargetAttribute(t, "TANGENT"); tan_it != primitive.targets[t].end()) {
                    const auto &tan_accessor = asset.accessors[tan_it->accessorIndex];
                    const auto prev_tan_size = morph->TangentDeltas.size() - target_count * prim_vertex_count;
                    if (tan_accessor.count == prim_vertex_count) {
                        fastgltf::copyFromAccessor<vec3>(asset, tan_accessor, &morph->TangentDeltas[prev_tan_size + t * prim_vertex_count]);
                    }
                }
            }
        }
    } else if (morph) {
        const uint32_t prim_vertex_count = position_accessor.count;
        morph->PositionDeltas.resize(morph->PositionDeltas.size() + morph->TargetCount * prim_vertex_count, vec3{0.f});
        if (!morph->NormalDeltas.empty()) {
            morph->NormalDeltas.resize(morph->NormalDeltas.size() + morph->TargetCount * prim_vertex_count, vec3{0.f});
        }
        if (!morph->TangentDeltas.empty()) {
            morph->TangentDeltas.resize(morph->TangentDeltas.size() + morph->TargetCount * prim_vertex_count, vec3{0.f});
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
        return std::unexpected{std::format("Failed to open glTF file '{}': no such file", path.string())};
    }
    auto gltf_file = fastgltf::MappedGltfFile::FromPath(path);
    if (gltf_file.error() != fastgltf::Error::None) return std::unexpected{std::format("Failed to open glTF file '{}': {}", path.string(), fastgltf::getErrorMessage(gltf_file.error()))};

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
                if (!missing.empty()) return std::unexpected{std::format("Failed to parse glTF '{}': Missing required extensions: {}", path.string(), missing)};
            }
        }
        return std::unexpected{std::format("Failed to parse glTF '{}': {}", path.string(), fastgltf::getErrorMessage(parsed.error()))};
    }

    auto &asset = parsed.get();
    if (auto decoded = DecodeMeshoptCompression(asset); !decoded) {
        return std::unexpected{std::format("Failed to decode meshopt compression in '{}': {}", path.string(), decoded.error())};
    }
    return std::move(asset);
}

// Preserves asset.meshes index alignment with an entry for empty meshes.
std::expected<uint32_t, std::string> EnsureMeshData(const fastgltf::Asset &asset, uint32_t source_mesh_index, std::vector<MeshData> &meshes, std::unordered_map<uint32_t, uint32_t> &mesh_index_map, size_t material_count) {
    if (const auto it = mesh_index_map.find(source_mesh_index); it != mesh_index_map.end()) return it->second;

    ::MeshData mesh;
    ::MeshVertexAttributes mesh_attrs;
    std::optional<ArmatureDeformData> mesh_deform;
    std::optional<MorphTargetData> mesh_morph;
    ::MeshData lines, points; // merged across all line/point primitives
    ::MeshVertexAttributes line_attrs, point_attrs;

    // Non-triangle primitives contribute 0 vertices here (their verts go into lines/points).
    const auto &source_mesh = asset.meshes[source_mesh_index];
    std::vector<uint32_t> vertex_counts(source_mesh.primitives.size(), 0);
    std::vector<uint32_t> attribute_flags(source_mesh.primitives.size(), 0);
    std::vector<uint8_t> has_source_indices(source_mesh.primitives.size(), 0);
    std::vector<std::vector<std::optional<uint32_t>>> variant_mappings(source_mesh.primitives.size());
    std::vector<uint32_t> face_primitive_indices;
    // Point and line primitives merge into one mesh each, so every vertex records its source primitive.
    std::vector<uint32_t> point_primitive_indices, line_primitive_indices;
    std::vector<uint32_t> primitive_material_indices(source_mesh.primitives.size(), material_count == 0 ? 0u : material_count - 1u);
    for (uint32_t primitive_index = 0; primitive_index < source_mesh.primitives.size(); ++primitive_index) {
        const auto &primitive = source_mesh.primitives[primitive_index];
        if (const auto material_index = ToIndex(primitive.materialIndex, material_count)) {
            primitive_material_indices[primitive_index] = *material_index;
        }
        has_source_indices[primitive_index] = primitive.indicesAccessor.has_value() ? 1u : 0u;
        if (!primitive.mappings.empty()) {
            auto &out = variant_mappings[primitive_index];
            out.reserve(primitive.mappings.size());
            for (const auto &m : primitive.mappings) {
                if (m.has_value()) out.emplace_back(ToIndex(*m, material_count));
                else out.emplace_back(std::nullopt);
            }
        }
        // Point and line shading keys off NORMAL and TANGENT, which the triangle append path records for itself.
        if (!IsTriangleType(primitive.type)) {
            if (primitive.findAttribute("NORMAL") != primitive.attributes.end()) attribute_flags[primitive_index] |= MeshAttributeBit_Normal;
            if (primitive.findAttribute("TANGENT") != primitive.attributes.end()) attribute_flags[primitive_index] |= MeshAttributeBit_Tangent;
        }
        // Point and line primitives append into their own mesh, recording each appended vertex's source primitive.
        const auto append_non_triangle = [&](::MeshData &data, ::MeshVertexAttributes &attrs, std::vector<uint32_t> &primitive_indices) {
            const auto prev_vertex_count = data.Positions.size();
            AppendNonTrianglePrimitive(asset, primitive, data, attrs);
            primitive_indices.insert(primitive_indices.end(), data.Positions.size() - prev_vertex_count, primitive_index);
        };
        if (primitive.type == fastgltf::PrimitiveType::Points) {
            append_non_triangle(points, point_attrs, point_primitive_indices);
            continue;
        }
        if (primitive.type == fastgltf::PrimitiveType::Lines || primitive.type == fastgltf::PrimitiveType::LineStrip || primitive.type == fastgltf::PrimitiveType::LineLoop) {
            append_non_triangle(lines, line_attrs, line_primitive_indices);
            continue;
        }
        const uint32_t prev_vertex_count = mesh.Positions.size(), prev_face_count = mesh.FaceCount();
        if (auto append_result = AppendPrimitive(asset, primitive, mesh, mesh_attrs, mesh_deform, mesh_morph, attribute_flags[primitive_index]); !append_result) {
            return std::unexpected{std::move(append_result.error())};
        }
        vertex_counts[primitive_index] = mesh.Positions.size() - prev_vertex_count;
        const auto appended_face_count = mesh.FaceCount() - prev_face_count;
        face_primitive_indices.insert(face_primitive_indices.end(), appended_face_count, primitive_index);
    }

    // Repack morph deltas from primitive-interleaved to per-target-contiguous.
    const auto triangle_prim_count = std::ranges::count_if(vertex_counts, [](auto c) { return c > 0; });
    if (mesh_morph && mesh_morph->TargetCount > 0 && triangle_prim_count > 1) {
        const uint32_t total_verts = mesh.Positions.size();
        const auto target_count = mesh_morph->TargetCount;
        const auto repack_channel = [&](std::vector<vec3> &channel) {
            if (channel.empty()) return;
            std::vector<vec3> repacked(target_count * total_verts, vec3{0.f});
            uint32_t src_off{0}, dst_vert_off{0};
            for (const auto prim_verts : vertex_counts) {
                if (prim_verts == 0) continue;
                for (uint32_t t = 0; t < target_count; ++t) {
                    for (uint32_t v = 0; v < prim_verts; ++v) {
                        repacked[t * total_verts + dst_vert_off + v] = channel[src_off + t * prim_verts + v];
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

    if (mesh_morph && !source_mesh.weights.empty()) {
        mesh_morph->DefaultWeights.resize(mesh_morph->TargetCount, 0.f);
        const auto copy_count = std::min(source_mesh.weights.size(), size_t(mesh_morph->TargetCount));
        std::copy_n(source_mesh.weights.begin(), copy_count, mesh_morph->DefaultWeights.begin());
    } else if (mesh_morph) {
        mesh_morph->DefaultWeights.assign(mesh_morph->TargetCount, 0.f);
    }

    const auto mesh_index = meshes.size();
    const bool has_triangles = !mesh.Positions.empty() && mesh.FaceCount() > 0;
    // Point and line meshes resolve their material per element through the same primitive-index tables the triangle mesh uses.
    const auto non_triangle_primitives = [&](std::vector<uint32_t> element_primitive_indices) {
        return element_primitive_indices.empty() ?
            ::MeshPrimitives{} :
            ::MeshPrimitives{std::move(element_primitive_indices), primitive_material_indices, attribute_flags, {}, {}};
    };
    auto line_primitives = non_triangle_primitives(std::move(line_primitive_indices));
    auto point_primitives = non_triangle_primitives(std::move(point_primitive_indices));
    meshes.emplace_back(MeshData{
        .Triangles = has_triangles ? std::optional{std::move(mesh)} : std::nullopt,
        .Lines = !lines.Positions.empty() ? std::optional{std::move(lines)} : std::nullopt,
        .Points = !points.Positions.empty() ? std::optional{std::move(points)} : std::nullopt,
        .TriangleAttrs = std::move(mesh_attrs),
        .LineAttrs = std::move(line_attrs),
        .PointAttrs = std::move(point_attrs),
        .TrianglePrimitives = has_triangles ? ::MeshPrimitives{std::move(face_primitive_indices), std::move(primitive_material_indices), std::move(attribute_flags), std::move(has_source_indices), std::move(variant_mappings)} : ::MeshPrimitives{},
        .LinePrimitives = std::move(line_primitives),
        .PointPrimitives = std::move(point_primitives),
        .DeformData = std::move(mesh_deform),
        .MorphData = std::move(mesh_morph),
        .Name = std::string{source_mesh.name},
    });
    mesh_index_map.emplace(source_mesh_index, mesh_index);
    return mesh_index;
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
bool AppendClip(entt::registry &r, entt::entity e, Clip &&clip) {
    if (clip.Channels.empty()) return false;
    if (auto *existing = r.try_get<Anim>(e)) existing->Clips.emplace_back(std::forward<Clip>(clip));
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

entt::entity ActiveSceneEntity(const entt::registry &r) {
    for (const auto e : r.view<const ActiveScene>()) return e;
    return entt::null;
}
// No SceneMembership (single-scene) means the node is in the sole scene, so always in the active one.
bool EntityInActiveScene(const entt::registry &r, entt::entity active_scene, entt::entity e) {
    const auto *sm = r.try_get<const SceneMembership>(e);
    return !sm || std::ranges::find(sm->Scenes, active_scene) != sm->Scenes.end();
}

// Toggle RenderInstance so only nodes in the active scene render. No-op for single-scene assets.
void ApplySceneVisibility(entt::registry &r) {
    const auto active = ActiveSceneEntity(r);
    for (auto [e, sm, _i] : r.view<const SceneMembership, const Instance>().each()) {
        if (std::ranges::find(sm.Scenes, active) != sm.Scenes.end()) Show(r, e);
        else Hide(r, e);
    }
}

// Selects an active imported entity by source order and camera, mesh, armature, root-empty, then object priority.
void ApplyActiveSceneSelection(entt::registry &r) {
    const auto active_scene = ActiveSceneEntity(r);

    // Armatures sort after source-indexed objects.
    std::vector<std::pair<uint32_t, entt::entity>> ordered;
    for (const auto e : r.view<const GltfObject, const ObjectKind>()) {
        if (EntityInActiveScene(r, active_scene, e)) {
            const auto *sni = r.try_get<const SourceNodeIndex>(e);
            ordered.emplace_back(sni ? sni->Value : std::numeric_limits<uint32_t>::max(), e);
        }
    }
    std::ranges::sort(ordered);

    const auto priority = [&](entt::entity e) {
        switch (r.get<const ObjectKind>(e).Value) {
            case ObjectType::Camera: return 0;
            case ObjectType::Mesh: return 1;
            case ObjectType::Armature: return 2;
            case ObjectType::Empty: return r.all_of<const SourceParentNodeIndex>(e) ? 4 : 3;
            default: return 4;
        }
    };
    entt::entity active = entt::null;
    int best = std::numeric_limits<int>::max();
    for (const auto &[_, e] : ordered) {
        if (const auto p = priority(e); p < best) {
            best = p;
            active = e;
        }
    }

    r.clear<Active, Selected>();
    if (active != entt::null) r.emplace<Active>(active);
    for (const auto &[_, e] : ordered) r.emplace<Selected>(e);
}
} // namespace

std::expected<fastgltf::Asset, std::string> ParseGltfAsset(const std::filesystem::path &path) { return ParseAsset(path); }

std::expected<LoadResult, std::string> LoadGltf(const std::filesystem::path &source_path, LoadContext ctx) {
    const profile::CpuScope scope{"LoadGltf"};

    ExtrasMap extras;
    auto parsed_asset = ParseAsset(source_path, &extras);
    if (!parsed_asset) return std::unexpected{parsed_asset.error()};

    auto &asset = *parsed_asset;
    if (asset.scenes.empty()) return std::unexpected{std::format("glTF '{}' has no scenes.", source_path.string())};

    const auto scene_index = asset.defaultScene.value_or(0);
    if (scene_index >= asset.scenes.size()) return std::unexpected{std::format("glTF '{}' has invalid default scene index.", source_path.string())};

    gltf::SourceAssets source_assets{
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
        .ImageBasedLight = {},
    };
    source_assets.Samplers.reserve(asset.samplers.size());
    for (const auto &sampler : asset.samplers) {
        source_assets.Samplers.emplace_back(Sampler{
            .MagFilter = ToFilter(sampler.magFilter),
            .MinFilter = ToFilter(sampler.minFilter),
            .WrapS = ToWrap(sampler.wrapS),
            .WrapT = ToWrap(sampler.wrapT),
            .Name = std::string{sampler.name},
        });
    }
    source_assets.Images.reserve(asset.images.size());
    const auto source_dir = AbsoluteScenePath(source_path).parent_path();
    for (uint32_t image_index = 0; image_index < asset.images.size(); ++image_index) {
        auto image_result = ReadImage(asset, image_index, source_dir);
        if (!image_result) return std::unexpected{std::move(image_result.error())};
        source_assets.Images.emplace_back(std::move(*image_result));
    }
    source_assets.Textures.reserve(asset.textures.size());
    for (const auto &texture : asset.textures) {
        source_assets.Textures.emplace_back(Texture{
            .SamplerIndex = ToIndex(texture.samplerIndex, asset.samplers.size()),
            .ImageIndex = ToIndex(texture.imageIndex, asset.images.size()),
            .WebpImageIndex = ToIndex(texture.webpImageIndex, asset.images.size()),
            .BasisuImageIndex = ToIndex(texture.basisuImageIndex, asset.images.size()),
            .DdsImageIndex = ToIndex(texture.ddsImageIndex, asset.images.size()),
            .Name = std::string{texture.name},
        });
    }

    // Store render materials and source metadata in parallel arrays with a trailing fallback entry.
    // Texture slots contain glTF indices until the emit loop maps them to bindless slots.
    std::vector<PBRMaterial> source_materials;
    source_materials.reserve(asset.materials.size() + 1u);
    std::vector<MaterialSourceMeta> material_metas;
    material_metas.reserve(asset.materials.size() + 1u);
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
        material_metas.emplace_back(std::move(meta));
        source_materials.emplace_back(std::move(pbr));
    }
    // Supply the required default for primitives without a material.
    source_materials.emplace_back();
    material_metas.emplace_back();

    const auto parents = BuildNodeParentTable(asset);
    std::vector<Transform> local_transforms(asset.nodes.size());
    std::vector<std::optional<mat4>> source_matrices(asset.nodes.size());
    for (uint32_t node_index = 0; node_index < asset.nodes.size(); ++node_index) {
        const auto &fg_transform = asset.nodes[node_index].transform;
        if (std::holds_alternative<fastgltf::TRS>(fg_transform)) {
            local_transforms[node_index] = TrsToTransform(std::get<fastgltf::TRS>(fg_transform));
        } else {
            const auto &fm = std::get<fastgltf::math::fmat4x4>(fg_transform);
            const mat4 m = std::bit_cast<mat4>(fm);
            source_matrices[node_index] = m;

            fastgltf::math::fvec3 scale, translation;
            fastgltf::math::fquat rotation;
            fastgltf::math::decomposeTransformMatrix(fm, scale, rotation, translation);
            local_transforms[node_index] = Transform{ToVec3(translation), numeric::Normalize(ToQuat(rotation)), ToVec3(scale)};
        }
    }
    // Process the default scene first to resolve shared-node hierarchy consistently.
    SceneTraversalData traversal{.InScene = std::vector(asset.nodes.size(), false), .WorldTransforms = std::vector(asset.nodes.size(), I4)};
    std::vector<uint32_t> node_to_scene_mask(asset.nodes.size(), 0u);
    const auto merge_scene = [&](uint32_t si) {
        const auto t = TraverseSceneNodes(asset, local_transforms, si);
        for (uint32_t i = 0; i < asset.nodes.size(); ++i) {
            if (!t.InScene[i]) continue;
            node_to_scene_mask[i] |= (1u << si);
            if (!traversal.InScene[i]) {
                traversal.InScene[i] = true;
                traversal.WorldTransforms[i] = t.WorldTransforms[i];
            }
        }
    };
    merge_scene(scene_index);
    for (uint32_t s = 0; s < asset.scenes.size(); ++s) {
        if (s != scene_index) merge_scene(s);
    }

    std::vector<bool> used_skin(asset.skins.size(), false);
    for (uint32_t node_index = 0; node_index < asset.nodes.size(); ++node_index) {
        if (!traversal.InScene[node_index]) continue;
        if (const auto skin_index = ToIndex(asset.nodes[node_index].skinIndex, asset.skins.size())) used_skin[*skin_index] = true;
    }

    // Parses KHR_physics_rigid_bodies document resources directly into entities.
    // Defers collision filters until the consumer block can use the shared name-deduplication map.
    std::vector<entt::entity> physics_material_entities, physics_jointdef_entities;
    {
        physics_material_entities.reserve(asset.physicsMaterials.size());
        for (uint32_t i = 0; i < asset.physicsMaterials.size(); ++i) {
            const auto &src = asset.physicsMaterials[i];
            const auto e = ctx.R.create();
            ctx.R.emplace<PhysicsMaterial>(e, PhysicsMaterial{.StaticFriction = src.staticFriction, .DynamicFriction = src.dynamicFriction, .Restitution = src.restitution, .FrictionCombine = ToCombineMode(src.frictionCombine), .RestitutionCombine = ToCombineMode(src.restitutionCombine)});
            ctx.R.emplace<SourcePhysicsMaterialIndex>(e, i);
            physics_material_entities.emplace_back(e);
        }
        physics_jointdef_entities.reserve(asset.physicsJoints.size());
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
            const auto e = ctx.R.create();
            ctx.R.emplace<PhysicsJointDef>(e, std::move(def));
            ctx.R.emplace<SourcePhysicsJointDefIndex>(e, i);
            physics_jointdef_entities.emplace_back(e);
        }
    }

    // Mesh-backed shapes resolve MeshEntity after node-to-entity mapping.
    const auto ToPhysicsShape = [&](const fastgltf::Geometry &geom) -> PhysicsShape {
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
    };

    std::vector<bool> is_bone(asset.nodes.size(), false);
    std::vector<std::optional<uint32_t>> skin_arma_node(asset.skins.size()); // Armature-root node per skin. Nullopt = scene root.
    std::vector<std::vector<uint32_t>> skin_joint_nodes(asset.skins.size()); // Valid joint nodes per used skin, deduped, source order.
    {
        std::vector<bool> node_carries_payload(asset.nodes.size(), false);
        for (uint32_t node_index = 0; node_index < asset.nodes.size(); ++node_index) {
            const auto &node = asset.nodes[node_index];
            node_carries_payload[node_index] = ToIndex(node.meshIndex, asset.meshes.size()).has_value() ||
                ToIndex(node.cameraIndex, asset.cameras.size()).has_value() ||
                ToIndex(node.lightIndex, asset.lights.size()).has_value() ||
                bool(node.physicsRigidBody) || !node.instancingAttributes.empty();
        }
        std::vector<std::optional<uint32_t>> skin_lca(asset.skins.size());
        for (uint32_t skin_index = 0; skin_index < asset.skins.size(); ++skin_index) {
            if (!used_skin[skin_index]) continue;
            const auto &skin = asset.skins[skin_index];
            auto &joint_nodes = skin_joint_nodes[skin_index];
            std::unordered_set<uint32_t> seen;
            for (const auto joint_idx : skin.joints) {
                if (const auto joint = ToIndex(joint_idx, asset.nodes.size()); joint && seen.emplace(*joint).second) joint_nodes.emplace_back(*joint);
            }
            if (joint_nodes.empty()) continue;
            auto lca_candidates = joint_nodes;
            if (const auto skel = ToIndex(skin.skeleton, asset.nodes.size())) lca_candidates.emplace_back(*skel);
            skin_lca[skin_index] = ComputeCommonAncestor(lca_candidates, parents);
            for (const auto joint : joint_nodes) is_bone[joint] = true;
        }
        for (bool changed = true; changed;) {
            changed = false;
            for (uint32_t skin_index = 0; skin_index < asset.skins.size(); ++skin_index) {
                if (skin_joint_nodes[skin_index].empty()) continue;
                auto arma = skin_lca[skin_index];
                while (arma && is_bone[*arma]) arma = parents[*arma];
                skin_arma_node[skin_index] = arma;
                for (const auto joint : skin_joint_nodes[skin_index]) {
                    for (std::optional<uint32_t> cur = joint; cur && cur != arma; cur = parents[*cur]) {
                        if (!is_bone[*cur] && !node_carries_payload[*cur]) {
                            is_bone[*cur] = true;
                            changed = true;
                        }
                    }
                }
            }
        }
    }

    // Preserve asset.meshes index alignment, including empty meshes.
    std::unordered_map<uint32_t, uint32_t> mesh_index_map;
    std::vector<MeshData> source_meshes;
    source_meshes.reserve(asset.meshes.size());
    for (uint32_t source_mesh_index = 0; source_mesh_index < asset.meshes.size(); ++source_mesh_index) {
        auto ensured = EnsureMeshData(asset, source_mesh_index, source_meshes, mesh_index_map, source_materials.size());
        if (!ensured) return std::unexpected{std::move(ensured.error())};
    }

    // Convert physics only for nodes carrying KHR_physics_rigid_bodies.
    // Other per-node info is read directly from asset.nodes/parents at the consumer site, so it needs no staging here.
    std::vector<NodePhysics> source_node_physics(asset.nodes.size());
    for (uint32_t node_index = 0; node_index < asset.nodes.size(); ++node_index) {
        const auto &source_node = asset.nodes[node_index];
        if (const auto &rb = source_node.physicsRigidBody) {
            auto &node = source_node_physics[node_index];
            if (rb->motion) {
                const auto com = ToVec3(rb->motion->centerOfMass);
                const auto inertia_orientation = rb->motion->inertialOrientation ? std::optional{std::bit_cast<quat>(*rb->motion->inertialOrientation)} : std::nullopt;
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
                const NodePhysics::MaterialRefs material{
                    .PhysicsMaterialIndex = ToIndex(rb->collider->physicsMaterial, asset.physicsMaterials.size()),
                    .CollisionFilterIndex = ToIndex(rb->collider->collisionFilter, asset.collisionFilters.size()),
                };
                if (material.PhysicsMaterialIndex || material.CollisionFilterIndex) node.Material = material;
                // source_meshes is index-aligned with asset.meshes, so the glTF mesh index is also the source mesh index.
                node.ColliderGeometryMeshIndex = ToIndex(rb->collider->geometry.mesh, asset.meshes.size());
            }
            if (rb->trigger) {
                NodePhysics::TriggerData trigger;
                std::visit(
                    [&](const auto &t) {
                        using T = std::decay_t<decltype(t)>;
                        if constexpr (std::is_same_v<T, fastgltf::GeometryTrigger>) {
                            trigger.Shape = ToPhysicsShape(t.geometry);
                            trigger.GeometryMeshIndex = ToIndex(t.geometry.mesh, asset.meshes.size());
                            trigger.CollisionFilterIndex = ToIndex(t.collisionFilter, asset.collisionFilters.size());
                        } else {
                            for (const auto n : t.nodes) {
                                if (n < asset.nodes.size()) trigger.NodeIndices.emplace_back(n);
                            }
                        }
                    },
                    *rb->trigger
                );
                node.Trigger = std::move(trigger);
            }
            if (rb->joint) {
                node.Joint = NodePhysics::JointData{
                    .ConnectedNodeIndex = uint32_t(rb->joint->connectedNode),
                    .JointDefIndex = uint32_t(rb->joint->joint),
                    .EnableCollision = rb->joint->enableCollision,
                };
            }
        }
    }

    std::vector<bool> is_object_emitted(asset.nodes.size(), false);
    for (uint32_t node_index = 0; node_index < asset.nodes.size(); ++node_index) {
        if (traversal.InScene[node_index]) {
            // Bone nodes render only when they also contain mesh data.
            const bool has_mesh = ToIndex(asset.nodes[node_index].meshIndex, asset.meshes.size()).has_value();
            is_object_emitted[node_index] = has_mesh || !is_bone[node_index];
        }
    }

    std::vector<std::optional<uint32_t>> nearest_object_ancestor(asset.nodes.size());
    for (uint32_t node_index = 0; node_index < asset.nodes.size(); ++node_index) {
        nearest_object_ancestor[node_index] = FindNearestMarkedAncestor(node_index, parents, is_object_emitted);
    }

    std::vector<Object> source_objects;
    for (uint32_t node_index = 0; node_index < asset.nodes.size(); ++node_index) {
        if (!is_object_emitted[node_index]) continue;
        const auto &source_node = asset.nodes[node_index];
        const auto mesh_index = ToIndex(source_node.meshIndex, asset.meshes.size());
        const auto skin_index = ToIndex(source_node.skinIndex, asset.skins.size());
        const auto camera_index = ToIndex(source_node.cameraIndex, asset.cameras.size());
        const auto light_index = ToIndex(source_node.lightIndex, asset.lights.size());
        const auto instance_transforms = traversal.InScene[node_index] ? ReadInstanceTransforms(asset, source_node) : std::vector<Transform>{};
        const auto &source_weights = source_node.weights;
        auto node_weights = source_weights.empty() ? std::optional<std::vector<float>>{} : std::optional{std::vector<float>(source_weights.begin(), source_weights.end())};
        if (!instance_transforms.empty() && mesh_index) {
            // EXT_mesh_gpu_instancing: emit one object per instance with baked world transform
            const auto base_name = MakeNodeName(asset, node_index, mesh_index);
            for (uint32_t i = 0; i < instance_transforms.size(); ++i) {
                // EXT_mesh_gpu_instancing: each instance is a root in the engine, so local == world.
                auto instance_world = ToTransform(traversal.WorldTransforms[node_index] * ToMatrix(instance_transforms[i]));
                source_objects.emplace_back(Object{
                    .ObjectType = Object::Type::Mesh,
                    .NodeIndex = node_index,
                    .ParentNodeIndex = std::nullopt,
                    .LocalTransform = instance_world,
                    .MeshIndex = mesh_index,
                    .SkinIndex = skin_index,
                    .CameraIndex = {},
                    .LightIndex = {},
                    .NodeWeights = node_weights,
                    .Name = base_name + "." + std::to_string(i),
                });
            }
        } else {
            const auto object_type = mesh_index ? Object::Type::Mesh :
                camera_index                    ? Object::Type::Camera :
                light_index                     ? Object::Type::Light :
                                                  Object::Type::Empty;
            source_objects.emplace_back(Object{
                .ObjectType = object_type,
                .NodeIndex = node_index,
                .ParentNodeIndex = nearest_object_ancestor[node_index],
                .LocalTransform = local_transforms[node_index],
                .MeshIndex = mesh_index,
                .SkinIndex = skin_index,
                .CameraIndex = camera_index,
                .LightIndex = light_index,
                .NodeWeights = std::move(node_weights),
                .Name = MakeNodeName(asset, node_index, mesh_index),
            });
        }
    }

    // Any skin with at least one valid joint reference and used_skin[i] is usable by the merged build/consume loop below.
    const bool any_usable_skin = [&] {
        for (uint32_t i = 0; i < asset.skins.size(); ++i) {
            if (!used_skin[i]) continue;
            for (const auto j : asset.skins[i].joints) {
                if (j < asset.nodes.size()) return true;
            }
        }
        return false;
    }();
    if (source_objects.empty() && !any_usable_skin) {
        return std::unexpected{std::format("glTF '{}' has no importable source objects or skins.", source_path.string())};
    }

    auto &r = ctx.R;
    const auto viewport = ctx.Viewport;
    auto &texture_store = ctx.Textures;
    const auto texture_start = texture_store.Textures.size();
    const auto material_start = ctx.Buffers.Materials.Count();
    const auto material_name_start = r.ctx().get<const MaterialStore>().Names.size();
    const auto pending_texture_start = r.all_of<PendingTextureUploads>(viewport) ? r.get<const PendingTextureUploads>(viewport).Items.size() : size_t{0};
    bool replaced_pending_env = false;
    std::optional<PendingEnvironmentImport> prev_pending_env_backup;
    const auto rollback_import_side_effects = [&] {
        if (texture_store.Textures.size() > texture_start) {
            ReleaseSamplerSlots(ctx.Slots, CollectSamplerSlots(std::span<const TextureEntry>{texture_store.Textures}.subspan(texture_start)));
            texture_store.Textures.resize(texture_start);
        }
        if (auto *pending = r.try_get<PendingTextureUploads>(viewport); pending && pending->Items.size() > pending_texture_start) {
            for (size_t i = pending_texture_start; i < pending->Items.size(); ++i) {
                ReleaseSamplerSlots(ctx.Slots, std::span{&pending->Items[i].SamplerSlot, 1});
            }
            pending->Items.resize(pending_texture_start);
            if (pending->Items.empty()) r.remove<PendingTextureUploads>(viewport);
        }
        if (replaced_pending_env) {
            if (auto *cur = r.try_get<PendingEnvironmentImport>(viewport)) {
                ReleaseCubeSamplerSlot(ctx.Slots, cur->DiffuseCubeSlot);
                ReleaseCubeSamplerSlot(ctx.Slots, cur->SpecularCubeSlot);
            }
            if (prev_pending_env_backup) r.emplace_or_replace<PendingEnvironmentImport>(viewport, std::move(*prev_pending_env_backup));
            else r.remove<PendingEnvironmentImport>(viewport);
        }
        if (ctx.Buffers.Materials.Count() > material_start) ctx.Buffers.Materials.SetCount(material_start);
        if (auto &store = r.ctx().get<MaterialStore>(); store.Names.size() > material_name_start) store.Names.resize(material_name_start);
    };
    struct ImportRollbackGuard {
        decltype(rollback_import_side_effects) &Rollback;
        bool Enabled{true};
        ~ImportRollbackGuard() {
            if (Enabled) Rollback();
        }
    };
    ImportRollbackGuard import_rollback_guard{rollback_import_side_effects};

    // Fill in the remaining SourceAssets fields (extensions, IBL, MaterialMetas), then emplace and bind a const ref.
    // The source-assets reference must outlive all subsequent load operations.
    auto source_ibl = ConvertIBL(asset, scene_index);
    source_assets.ImageBasedLight = source_ibl;
    source_assets.ExtensionsRequired.reserve(asset.extensionsRequired.size());
    for (const auto &e : asset.extensionsRequired) source_assets.ExtensionsRequired.emplace_back(e);
    source_assets.MaterialMetas = std::move(material_metas);
    const auto &sa = r.emplace_or_replace<gltf::SourceAssets>(viewport, std::move(source_assets));

    if (!asset.materialVariants.empty()) {
        ::MaterialVariants mv;
        mv.Names.reserve(asset.materialVariants.size());
        for (const auto &v : asset.materialVariants) mv.Names.emplace_back(v);
        r.emplace_or_replace<::MaterialVariants>(viewport, std::move(mv));
    } else {
        r.remove<::MaterialVariants>(viewport);
    }

    std::vector<PendingTextureUpload> new_pending_textures;
    std::unordered_map<uint64_t, uint32_t> texture_slot_cache;
    // Caches by resolved image, sampler, and color space so equivalent glTF textures share one TextureEntry.
    const auto texture_cache_key = [](uint32_t image_index, uint32_t sampler_index, TextureColorSpace color_space) {
        return (uint64_t(image_index) << 33u) | (uint64_t(sampler_index) << 1u) | (color_space == TextureColorSpace::Srgb ? 1u : 0u);
    };
    const auto resolve_texture_slot = [&](uint32_t texture_index, TextureColorSpace color_space) -> std::expected<uint32_t, std::string> {
        if (texture_index >= sa.Textures.size()) return InvalidSlot;

        const auto &src_texture = sa.Textures[texture_index];
        const auto image_index = gltf::ResolveImageIndex(src_texture);
        if (!image_index || *image_index >= sa.Images.size()) return InvalidSlot;

        const auto sampler_index = src_texture.SamplerIndex.value_or(InvalidSlot);
        const auto cache_key = texture_cache_key(*image_index, sampler_index, color_space);
        if (const auto it = texture_slot_cache.find(cache_key); it != texture_slot_cache.end()) return it->second;

        const auto *src_sampler = src_texture.SamplerIndex && *src_texture.SamplerIndex < sa.Samplers.size() ?
            &sa.Samplers[*src_texture.SamplerIndex] :
            nullptr;
        static constexpr auto ToSamplerAddressMode = [](gltf::Wrap wrap) {
            switch (wrap) {
                case gltf::Wrap::ClampToEdge: return MTL::SamplerAddressModeClampToEdge;
                case gltf::Wrap::MirroredRepeat: return MTL::SamplerAddressModeMirrorRepeat;
                case gltf::Wrap::Repeat: return MTL::SamplerAddressModeRepeat;
            }
            return MTL::SamplerAddressModeRepeat;
        };
        static constexpr auto ToSamplerConfig = [](const gltf::Sampler *sampler) -> SamplerConfig {
            if (!sampler) return {.MinFilter = MTL::SamplerMinMagFilterLinear, .MagFilter = MTL::SamplerMinMagFilterLinear, .MipmapMode = MTL::SamplerMipFilterLinear, .UsesMipmaps = true};

            const auto mag_filter = sampler->MagFilter && *sampler->MagFilter == gltf::Filter::Nearest ? MTL::SamplerMinMagFilterNearest : MTL::SamplerMinMagFilterLinear;
            switch (sampler->MinFilter.value_or(gltf::Filter::LinearMipMapLinear)) {
                case gltf::Filter::Nearest:
                    return {.MinFilter = MTL::SamplerMinMagFilterNearest, .MagFilter = mag_filter, .MipmapMode = MTL::SamplerMipFilterNearest, .UsesMipmaps = false};
                case gltf::Filter::Linear:
                    return {.MinFilter = MTL::SamplerMinMagFilterLinear, .MagFilter = mag_filter, .MipmapMode = MTL::SamplerMipFilterNearest, .UsesMipmaps = false};
                case gltf::Filter::NearestMipMapNearest:
                    return {.MinFilter = MTL::SamplerMinMagFilterNearest, .MagFilter = mag_filter, .MipmapMode = MTL::SamplerMipFilterNearest, .UsesMipmaps = true};
                case gltf::Filter::LinearMipMapNearest:
                    return {.MinFilter = MTL::SamplerMinMagFilterLinear, .MagFilter = mag_filter, .MipmapMode = MTL::SamplerMipFilterNearest, .UsesMipmaps = true};
                case gltf::Filter::NearestMipMapLinear:
                    return {.MinFilter = MTL::SamplerMinMagFilterNearest, .MagFilter = mag_filter, .MipmapMode = MTL::SamplerMipFilterLinear, .UsesMipmaps = true};
                case gltf::Filter::LinearMipMapLinear:
                    return {.MinFilter = MTL::SamplerMinMagFilterLinear, .MagFilter = mag_filter, .MipmapMode = MTL::SamplerMipFilterLinear, .UsesMipmaps = true};
            }
            return {.MinFilter = MTL::SamplerMinMagFilterLinear, .MagFilter = mag_filter, .MipmapMode = MTL::SamplerMipFilterLinear, .UsesMipmaps = true};
        };

        const auto sampler_config = ToSamplerConfig(src_sampler);
        const auto wrap_s = src_sampler ? ToSamplerAddressMode(src_sampler->WrapS) : MTL::SamplerAddressModeRepeat;
        const auto wrap_t = src_sampler ? ToSamplerAddressMode(src_sampler->WrapT) : MTL::SamplerAddressModeRepeat;
        auto texture_name = std::format("{} ({})", src_texture.Name.empty() ? std::format("Texture{}", texture_index) : src_texture.Name, color_space == TextureColorSpace::Srgb ? "sRGB" : "Linear");

        const auto sampler_slot = AllocateSamplerSlot(ctx.Slots);
        new_pending_textures.emplace_back(PendingTextureUpload{
            .SamplerSlot = sampler_slot,
            .Source = PendingTextureUpload::GltfImageRef{*image_index},
            .ColorSpace = color_space,
            .WrapS = wrap_s,
            .WrapT = wrap_t,
            .Sampler = sampler_config,
            .Name = std::move(texture_name),
        });
        texture_slot_cache.emplace(cache_key, sampler_slot);
        return sampler_slot;
    };

    std::vector<uint32_t> material_indices_by_gltf_material(source_materials.size(), 0u);
    const auto material_count = ctx.Buffers.Materials.Count();
    const auto default_material_index = material_count > 0 ? material_count - 1u : 0u;
    std::vector<std::string> material_names;
    material_names.reserve(source_materials.size());
    ctx.Buffers.Materials.ReserveElements(material_count + source_materials.size());
    for (uint32_t material_index = 0; material_index < source_materials.size(); ++material_index) {
        const auto &src_material = source_materials[material_index];
        const auto src_name = material_index < asset.materials.size() ? std::string_view(asset.materials[material_index].name) : std::string_view{"DefaultMaterial"};
        const auto material_name = src_name.empty() ? std::format("Material{}", material_index) : std::string{src_name};
        const auto clamp_uv_set = [&](uint32_t uv_set, std::string_view texture_label) {
            if (uv_set <= 3u) return uv_set;
            std::cerr << std::format(
                "Warning: glTF material '{}' texture '{}' uses TEXCOORD_{}. MeshEditor currently supports TEXCOORD_0..3. Clamping to TEXCOORD_3.\n",
                material_name, texture_label, uv_set
            );
            return 3u;
        };
        // Replaces a GltfLoader texture index in tex.Slot with its bindless sampler slot.
        // UV fields remain unchanged.
        const auto resolve_texture = [&](TextureInfo &tex, TextureColorSpace color_space, std::string_view texture_label) -> std::expected<void, std::string> {
            if (tex.Slot == InvalidSlot) return {};
            const uint32_t gltf_index = tex.Slot;
            tex.TexCoord = clamp_uv_set(tex.TexCoord, texture_label);
            auto texture_slot_result = resolve_texture_slot(gltf_index, color_space);
            if (!texture_slot_result) return std::unexpected{std::move(texture_slot_result.error())};
            tex.Slot = *texture_slot_result;
            return {};
        };
        auto gpu_material = src_material;
        for (const auto &slot : MaterialTextureSlots) {
            if (auto result = resolve_texture(slot.Get(gpu_material), slot.ColorSpace, slot.Label); !result) {
                return std::unexpected{std::move(result.error())};
            }
        }
        material_indices_by_gltf_material[material_index] = ctx.Buffers.Materials.Append(gpu_material);
        material_names.emplace_back(material_name);
    }
    const auto fallback_material_index = material_indices_by_gltf_material.empty() ? default_material_index : material_indices_by_gltf_material.back();
    // Map a source gltf material index to its post-load PrimitiveMaterialBuffer index.
    const auto remap_material = [&](uint32_t i) { return i < material_indices_by_gltf_material.size() ? material_indices_by_gltf_material[i] : fallback_material_index; };
    if (!material_names.empty()) {
        auto &store = r.ctx().get<MaterialStore>();
        store.Names.insert(store.Names.end(), std::make_move_iterator(material_names.begin()), std::make_move_iterator(material_names.end()));
    }

    // Capture resolved texture uploads before moving them so snapshots can reuse their bindless slots.
    // Emplaced only on success, after the rollback guard is disarmed.
    std::vector<MaterializedTexture> materialized_textures;
    materialized_textures.reserve(new_pending_textures.size());
    for (const auto &t : new_pending_textures) {
        materialized_textures.emplace_back(MaterializedTexture{
            .SamplerSlot = t.SamplerSlot,
            .SourceImageIndex = std::get<PendingTextureUpload::GltfImageRef>(t.Source).ImageIndex,
            .ColorSpace = t.ColorSpace,
            .WrapS = t.WrapS,
            .WrapT = t.WrapT,
            .Sampler = t.Sampler,
            .Name = t.Name,
        });
    }
    if (!new_pending_textures.empty()) {
        auto &pending = r.get_or_emplace<PendingTextureUploads>(viewport);
        pending.Items.insert(pending.Items.end(), std::make_move_iterator(new_pending_textures.begin()), std::make_move_iterator(new_pending_textures.end()));
    }

    struct MorphSummary {
        uint32_t TargetCount{};
        std::vector<float> DefaultWeights;
    };
    struct NonTriangleEntities {
        entt::entity Lines{entt::null}, Points{entt::null};
    };
    // Maps each source mesh to its triangle, line, and point batch entries.
    static constexpr uint32_t NoPart{UINT32_MAX};
    struct SourceParts {
        uint32_t Triangles{NoPart}, Lines{NoPart}, Points{NoPart};
    };

    std::vector<MeshSource> sources;
    std::vector<MeshSourceLayout> layouts;
    std::vector<SourceParts> parts(source_meshes.size());
    std::vector<PbrFeatureMask> pbr_masks(source_meshes.size(), PbrFeatureMask{0});
    std::vector<MorphSummary> mesh_morphs(source_meshes.size());
    for (uint32_t mi = 0; mi < source_meshes.size(); ++mi) {
        auto &scene_mesh = source_meshes[mi];
        if (scene_mesh.Triangles) {
            // Must run before the remap loop below overwrites MaterialIndices, since this indexes source_materials by gltf index.
            // The texture tests only check for InvalidSlot, which survives slot remapping, so reading pre-remap slots is fine.
            PbrFeatureMask mesh_pbr_mask{0};
            for (const auto gltf_mat_idx : scene_mesh.TrianglePrimitives.MaterialIndices) {
                if (gltf_mat_idx < source_materials.size()) {
                    const auto &mat = source_materials[gltf_mat_idx];
                    if (mat.Transmission.Factor > 0.f || mat.Transmission.Texture.Slot != InvalidSlot) mesh_pbr_mask |= PbrFeature::Transmission;
                    if (mat.DiffuseTransmission.Factor > 0.f || mat.DiffuseTransmission.Texture.Slot != InvalidSlot) mesh_pbr_mask |= PbrFeature::DiffuseTrans;
                    if (mat.Clearcoat.Factor > 0.f || mat.Clearcoat.Texture.Slot != InvalidSlot) mesh_pbr_mask |= PbrFeature::Clearcoat;
                    if (mat.Sheen.RoughnessFactor > 0.f || mat.Sheen.ColorTexture.Slot != InvalidSlot) mesh_pbr_mask |= PbrFeature::Sheen;
                    if (mat.Anisotropy.Strength != 0.f || mat.Anisotropy.Texture.Slot != InvalidSlot) mesh_pbr_mask |= PbrFeature::Anisotropy;
                    if (mat.Iridescence.Factor > 0.f || mat.Iridescence.Texture.Slot != InvalidSlot) mesh_pbr_mask |= PbrFeature::Iridescence;
                }
            }
            pbr_masks[mi] = mesh_pbr_mask;
            for (auto &local_material_index : scene_mesh.TrianglePrimitives.MaterialIndices) local_material_index = remap_material(local_material_index);
            // KHR_materials_variants mappings contain source glTF material indices.
            // Remap to match post-load PrimitiveMaterialBuffer indices so the runtime can apply a variant by writing entries straight to that buffer.
            for (auto &prim_mappings : scene_mesh.TrianglePrimitives.VariantMappings) {
                for (auto &m : prim_mappings) {
                    if (m) *m = remap_material(*m);
                }
            }
            // Snapshot per-primitive metadata before the batch consumes the source.
            // DefaultMaterials copies because the create also consumes MaterialIndices to populate PrimitiveMaterialBuffer.
            MeshSourceLayout layout{
                .AttributeFlags = std::move(scene_mesh.TrianglePrimitives.AttributeFlags),
                .HasSourceIndices = std::move(scene_mesh.TrianglePrimitives.HasSourceIndices),
                .DefaultMaterials = scene_mesh.TrianglePrimitives.MaterialIndices,
                .VariantMappings = std::move(scene_mesh.TrianglePrimitives.VariantMappings),
                .Colors0ComponentCount = scene_mesh.TriangleAttrs.Colors0ComponentCount,
                .MorphTangentDeltas = {}, // Filled from the created mesh below, after welding compacts them.
            };
            if (scene_mesh.MorphData) mesh_morphs[mi] = {scene_mesh.MorphData->TargetCount, scene_mesh.MorphData->DefaultWeights};
            // Primitives without NORMAL are flat-shaded per the glTF spec.
            const bool any_normals = std::ranges::any_of(layout.AttributeFlags, [](uint32_t flags) { return (flags & MeshAttributeBit_Normal) != 0; });
            parts[mi].Triangles = uint32_t(sources.size());
            sources.emplace_back(MeshSource{
                .Data = std::move(*scene_mesh.Triangles),
                .Attrs = std::move(scene_mesh.TriangleAttrs),
                .Primitives = std::move(scene_mesh.TrianglePrimitives),
                .Deform = std::move(scene_mesh.DeformData),
                .Morph = std::move(scene_mesh.MorphData),
                .Weld = true,
                .FlatShaded = !any_normals,
            });
            layouts.emplace_back(std::move(layout));
        }
        // Point and line primitives draw straight from their positions, so they record no source index streams.
        const auto add_non_triangle = [&](std::optional<::MeshData> &data, ::MeshVertexAttributes &attrs, ::MeshPrimitives &primitives) {
            if (!data) return NoPart;
            for (auto &local_material_index : primitives.MaterialIndices) local_material_index = remap_material(local_material_index);
            layouts.emplace_back(MeshSourceLayout{
                .AttributeFlags = primitives.AttributeFlags,
                .HasSourceIndices = {},
                .DefaultMaterials = primitives.MaterialIndices,
                .VariantMappings = {},
                .Colors0ComponentCount = attrs.Colors0ComponentCount,
                .MorphTangentDeltas = {},
            });
            const auto part = uint32_t(sources.size());
            sources.emplace_back(MeshSource{.Data = std::move(*data), .Attrs = std::move(attrs), .Primitives = std::move(primitives)});
            return part;
        };
        parts[mi].Lines = add_non_triangle(scene_mesh.Lines, scene_mesh.LineAttrs, scene_mesh.LinePrimitives);
        parts[mi].Points = add_non_triangle(scene_mesh.Points, scene_mesh.PointAttrs, scene_mesh.PointPrimitives);
    }

    // Derive the batch in source order.
    auto created = CreateMeshes(r, sources);

    std::vector<entt::entity> mesh_entities;
    mesh_entities.reserve(source_meshes.size());
    std::vector<NonTriangleEntities> non_triangle_entities_per_mesh(source_meshes.size());
    for (uint32_t mi = 0; mi < source_meshes.size(); ++mi) {
        const auto &scene_mesh = source_meshes[mi];
        const auto add_part = [&](uint32_t part, MeshKind kind) {
            auto &layout = layouts[part];
            layout.MorphTangentDeltas = std::move(created[part].MorphTangentDeltas);
            const auto [e, _] = ::AddMesh(r, created[part].StoreId, std::nullopt);
            r.emplace<Path>(e, source_path);
            r.emplace<SourceMeshIndex>(e, mi);
            r.emplace<SourceMeshKind>(e, kind);
            r.emplace<MeshSourceLayout>(e, std::move(layout));
            if (!scene_mesh.Name.empty()) r.emplace<MeshName>(e, scene_mesh.Name);
            return e;
        };
        entt::entity mesh_entity = entt::null;
        if (parts[mi].Triangles != NoPart) {
            mesh_entity = add_part(parts[mi].Triangles, MeshKind::Triangles);
            if (pbr_masks[mi] != 0) r.emplace<PbrMeshFeatures>(mesh_entity, pbr_masks[mi]);
        }
        mesh_entities.emplace_back(mesh_entity);
        non_triangle_entities_per_mesh[mi] = {
            parts[mi].Lines == NoPart ? entt::null : add_part(parts[mi].Lines, MeshKind::Lines),
            parts[mi].Points == NoPart ? entt::null : add_part(parts[mi].Points, MeshKind::Points),
        };
    }

    const auto name_prefix = source_path.stem().string();
    ReserveEntityNames(r, source_objects.size());
    std::unordered_map<uint32_t, entt::entity> object_entities_by_node;
    object_entities_by_node.reserve(source_objects.size());
    std::unordered_map<uint32_t, std::vector<entt::entity>> skinned_mesh_instances_by_skin;
    skinned_mesh_instances_by_skin.reserve(asset.skins.size());
    std::vector<entt::entity> armature_data_entities;

    entt::entity first_object_entity = entt::null,
                 first_mesh_object_entity = entt::null,
                 first_camera_object_entity = entt::null,
                 first_root_empty_entity = entt::null,
                 first_armature_entity = entt::null;
    for (uint32_t i = 0; i < source_objects.size(); ++i) {
        const auto &object = source_objects[i];
        const auto object_name = object.Name.empty() ? std::format("{}_{}", name_prefix, i) : object.Name;
        entt::entity object_entity = entt::null;
        // Prefer Triangles, then Lines, then Points (for Lines/Points-only source meshes).
        const auto primary_mesh_entity = [&]() -> entt::entity {
            if (object.ObjectType != gltf::Object::Type::Mesh || !object.MeshIndex) return entt::null;
            const auto mi = *object.MeshIndex;
            if (mi < mesh_entities.size() && mesh_entities[mi] != entt::null) return mesh_entities[mi];
            if (mi < non_triangle_entities_per_mesh.size()) {
                const auto &[lines, points] = non_triangle_entities_per_mesh[mi];
                return lines != entt::null ? lines : points;
            }
            return entt::null;
        }();
        if (primary_mesh_entity != entt::null) {
            object_entity = ::AddMeshInstance(
                r,
                primary_mesh_entity,
                {.Name = object_name, .Transform = object.LocalTransform, .Select = MeshInstanceCreateInfo::SelectBehavior::None, .Visible = true}
            );
        } else if (object.ObjectType == gltf::Object::Type::Camera && object.CameraIndex && *object.CameraIndex < asset.cameras.size()) {
            const auto &cam = asset.cameras[*object.CameraIndex];
            object_entity = ::AddCamera(r, ctx.Meshes, {.Name = object_name, .Transform = object.LocalTransform, .Select = MeshInstanceCreateInfo::SelectBehavior::None});
            r.replace<::Camera>(object_entity, ConvertCamera(cam));
            r.emplace<SourceCameraIndex>(object_entity, *object.CameraIndex);
            if (!cam.name.empty()) r.emplace<CameraName>(object_entity, std::string{cam.name});
        } else if (object.ObjectType == gltf::Object::Type::Light && object.LightIndex && *object.LightIndex < asset.lights.size()) {
            const auto &light = asset.lights[*object.LightIndex];
            object_entity = ::AddLight(r, ctx.Meshes, {.Name = object_name, .Transform = object.LocalTransform, .Select = MeshInstanceCreateInfo::SelectBehavior::None}, ConvertLight(light));
            r.emplace<SourceLightIndex>(object_entity, *object.LightIndex);
            if (!light.name.empty()) r.emplace<LightName>(object_entity, std::string{light.name});
        } else {
            object_entity = ::AddEmpty(r, ctx.Meshes, {.Name = object_name, .Transform = object.LocalTransform, .Select = MeshInstanceCreateInfo::SelectBehavior::None});
        }
        // Parent non-triangle instances under the primary instance with identity transforms.
        if (object.ObjectType == gltf::Object::Type::Mesh && object.MeshIndex && *object.MeshIndex < non_triangle_entities_per_mesh.size()) {
            const auto &non_triangle = non_triangle_entities_per_mesh[*object.MeshIndex];
            for (const auto extra_entity : {non_triangle.Lines, non_triangle.Points}) {
                if (extra_entity != entt::null && extra_entity != primary_mesh_entity) {
                    const auto extra_instance = ::AddMeshInstance(
                        r,
                        extra_entity,
                        {.Name = object_name, .Transform = Transform{}, .Select = MeshInstanceCreateInfo::SelectBehavior::None, .Visible = true}
                    );
                    SetParent(r, extra_instance, object_entity);
                }
            }
        }

        object_entities_by_node[object.NodeIndex] = object_entity;
        r.emplace<SourceNodeIndex>(object_entity, object.NodeIndex);
        // Compare synthesized object.Name with the raw source name to record empty or collision-renamed values.
        if (object.NodeIndex < asset.nodes.size()) {
            const std::string raw_name(asset.nodes[object.NodeIndex].name);
            if (raw_name.empty()) r.emplace<SourceEmptyName>(object_entity);
            else if (const auto *n = r.try_get<const Name>(object_entity); n && n->Value != raw_name) {
                r.emplace<SourceObjectName>(object_entity, SourceObjectName{raw_name});
            }
        }
        r.emplace<GltfObject>(object_entity);
        // glTF node.skin is deform linkage, not a transform-parent relationship.
        if (object.SkinIndex && r.all_of<Instance>(object_entity)) skinned_mesh_instances_by_skin[*object.SkinIndex].emplace_back(object_entity);
        if (first_object_entity == entt::null) first_object_entity = object_entity;
        if (first_mesh_object_entity == entt::null && object.ObjectType == gltf::Object::Type::Mesh) first_mesh_object_entity = object_entity;
        if (first_camera_object_entity == entt::null && object.ObjectType == gltf::Object::Type::Camera) first_camera_object_entity = object_entity;
        if (first_root_empty_entity == entt::null && object.ObjectType == gltf::Object::Type::Empty && !object.ParentNodeIndex) first_root_empty_entity = object_entity;
    }

    for (const auto &object : source_objects) {
        if (!object.ParentNodeIndex) continue;

        const auto child_it = object_entities_by_node.find(object.NodeIndex);
        if (child_it == object_entities_by_node.end()) continue;
        const auto parent_it = object_entities_by_node.find(*object.ParentNodeIndex);
        if (parent_it != object_entities_by_node.end()) {
            SetParent(r, child_it->second, parent_it->second);
        }
    }

    // Create serialization-only stubs for nodes referenced exclusively by non-default scenes.
    for (uint32_t node_index = 0; node_index < asset.nodes.size(); ++node_index) {
        if (traversal.InScene[node_index]) continue;
        const auto &source_node = asset.nodes[node_index];
        const auto e = r.create();
        r.emplace<SourceNodeIndex>(e, node_index);
        r.emplace<Transform>(e, local_transforms[node_index]);
        r.emplace<WorldTransform>(e);
        if (const auto mesh_index = ToIndex(source_node.meshIndex, asset.meshes.size());
            mesh_index && *mesh_index < mesh_entities.size() && mesh_entities[*mesh_index] != entt::null) {
            r.emplace<Instance>(e, mesh_entities[*mesh_index]);
        }
        if (source_node.name.empty()) {
            r.emplace<SourceEmptyName>(e);
        } else {
            const std::string raw_name{source_node.name};
            const auto &name = EmplaceUniqueName(r, e, raw_name);
            if (name.Value != raw_name) r.emplace<SourceObjectName>(e, raw_name);
        }
    }

    // Creates collision-filter entities with one system-name deduplication map shared across all filters.
    {
        // Dedupe system names across all filters into CollisionSystem entities.
        std::unordered_map<std::string, entt::entity> system_entity_by_name;
        const auto resolve_systems = [&](const auto &names) {
            std::vector<entt::entity> out;
            out.reserve(names.size());
            for (const auto &n : names) {
                std::string key{n};
                auto [it, inserted] = system_entity_by_name.try_emplace(std::move(key), entt::null);
                if (inserted) {
                    it->second = r.create();
                    r.emplace<CollisionSystem>(it->second, CollisionSystem{.Name = it->first});
                }
                out.emplace_back(it->second);
            }
            return out;
        };

        std::vector<entt::entity> filter_entities;
        filter_entities.reserve(asset.collisionFilters.size());
        for (uint32_t i = 0; i < asset.collisionFilters.size(); ++i) {
            const auto &src = asset.collisionFilters[i];
            // KHR schema forbids both collideWith and notCollideWith; prefer allowlist if both appear.
            auto [mode, collide_systems] = [&]() -> std::pair<CollideMode, std::vector<entt::entity>> {
                if (!src.collideWithSystems.empty()) return {CollideMode::Allowlist, resolve_systems(src.collideWithSystems)};
                if (!src.notCollideWithSystems.empty()) return {CollideMode::Blocklist, resolve_systems(src.notCollideWithSystems)};
                return {CollideMode::All, {}};
            }();
            const auto e = r.create();
            r.emplace<CollisionFilter>(e, CollisionFilter{.Systems = resolve_systems(src.collisionSystems), .Mode = mode, .CollideSystems = std::move(collide_systems)});
            r.emplace<SourceCollisionFilterIndex>(e, i);
            filter_entities.emplace_back(e);
        }

        auto resolve_mat = [&](std::optional<uint32_t> idx) {
            return idx && *idx < physics_material_entities.size() ? physics_material_entities[*idx] : null_entity;
        };
        auto resolve_filter = [&](std::optional<uint32_t> idx) {
            return idx && *idx < filter_entities.size() ? filter_entities[*idx] : null_entity;
        };

        for (uint32_t node_index = 0; node_index < source_node_physics.size(); ++node_index) {
            const auto &node = source_node_physics[node_index];
            auto it = object_entities_by_node.find(node_index);
            if (it == object_entities_by_node.end()) continue;
            const auto entity = it->second;

            if (node.Collider) {
                const auto collider_mesh_entity = [&]() -> entt::entity {
                    if (!IsMeshBackedShape(node.Collider->Shape)) return null_entity;
                    if (node.ColliderGeometryMeshIndex && *node.ColliderGeometryMeshIndex < mesh_entities.size()) {
                        return mesh_entities[*node.ColliderGeometryMeshIndex];
                    }
                    if (r.all_of<Instance>(entity)) return r.get<const Instance>(entity).Entity;
                    return null_entity;
                }();
                r.emplace<ColliderShape>(entity, ColliderShape{.Shape = node.Collider->Shape, .MeshEntity = collider_mesh_entity});
                // Imported collider state is authoritative — engine must not auto-derive over it.
                r.emplace<ColliderPolicy>(entity, ColliderPolicy{.AutoFitDims = false, .LockedKind = true});
                if (node.Material) {
                    r.replace<ColliderMaterial>(
                        entity,
                        ColliderMaterial{
                            .PhysicsMaterialEntity = resolve_mat(node.Material->PhysicsMaterialIndex),
                            .CollisionFilterEntity = resolve_filter(node.Material->CollisionFilterIndex),
                        }
                    );
                }
            }
            if (node.Motion) {
                r.emplace<PhysicsMotion>(entity, *node.Motion);
                if (node.Velocity) r.replace<PhysicsVelocity>(entity, *node.Velocity);
            }
            if (node.Trigger) {
                const auto &td = *node.Trigger;
                if (td.Shape) {
                    // Represents GeometryTrigger with ColliderShape and TriggerTag.
                    // Skip entities already used by a solid collider because KHR makes the two forms exclusive.
                    if (!r.all_of<ColliderShape>(entity)) {
                        const auto trigger_mesh_entity = (td.GeometryMeshIndex && *td.GeometryMeshIndex < mesh_entities.size()) ? mesh_entities[*td.GeometryMeshIndex] : null_entity;
                        r.emplace<ColliderShape>(entity, ColliderShape{.Shape = *td.Shape, .MeshEntity = trigger_mesh_entity});
                        r.emplace<ColliderPolicy>(entity, ColliderPolicy{.AutoFitDims = false, .LockedKind = true});
                        r.emplace<TriggerTag>(entity);
                        r.patch<ColliderMaterial>(entity, [&](auto &m) { m.CollisionFilterEntity = resolve_filter(td.CollisionFilterIndex); });
                    }
                } else {
                    // NodesTrigger: compound zone.
                    std::vector<entt::entity> resolved_nodes;
                    resolved_nodes.reserve(td.NodeIndices.size());
                    for (const auto node_idx : td.NodeIndices) {
                        auto nit = object_entities_by_node.find(node_idx);
                        resolved_nodes.emplace_back(nit != object_entities_by_node.end() ? nit->second : entt::null);
                    }
                    r.emplace<TriggerNodes>(entity, TriggerNodes{.Nodes = std::move(resolved_nodes), .CollisionFilterEntity = resolve_filter(td.CollisionFilterIndex)});
                }
            }
            if (node.Joint) {
                const auto &jd = *node.Joint;
                auto nit = object_entities_by_node.find(jd.ConnectedNodeIndex);
                const auto def_entity = jd.JointDefIndex < physics_jointdef_entities.size() ? physics_jointdef_entities[jd.JointDefIndex] : null_entity;
                r.emplace<PhysicsJoint>(
                    entity,
                    PhysicsJoint{.ConnectedNode = nit != object_entities_by_node.end() ? nit->second : entt::null, .JointDefEntity = def_entity, .EnableCollision = jd.EnableCollision}
                );
            }
        }
    }

    // KHR_audio_rigid_bodies: rebuild modal models, acoustic materials, and acoustic surfaces, attaching them to each instancing node and its mesh entity.
    if (!asset.modalModels.empty() || !asset.acousticSurfaces.empty()) {
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
            if (!source_node.audioRigidBody.has_value()) continue;
            const auto &instance = *source_node.audioRigidBody;
            const auto it = object_entities_by_node.find(node_index);
            if (it == object_entities_by_node.end()) continue;
            const auto entity = it->second;

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
                r.emplace<MassProperties>(
                    entity,
                    MassProperties{
                        .Mass = mp->mass,
                        .CenterOfMass = ToVec3(mp->centerOfMass),
                        .InertiaDiagonal = ToVec3(mp->inertiaDiagonal),
                        .InertiaOrientation = std::bit_cast<quat>(q),
                    }
                );
                // Uses dynamic rigid-body mass for sound contact dynamics; see UpdateContactDynamics.
                // Warn when modal and rigid-body masses differ.
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
                r.emplace<MassProperties>(
                    entity,
                    MassProperties{
                        .Mass = *motion->Mass,
                        .CenterOfMass = motion->CenterOfMass.value_or(vec3{0}),
                        .InertiaDiagonal = motion->InertiaDiagonal.value_or(vec3{0}),
                        .InertiaOrientation = motion->InertiaOrientation.value_or(quat{1, 0, 0, 0}),
                    }
                );
            }
            if (excitable) r.emplace<SoundVerticesModel>(entity, SoundVerticesModel::Modal);
            if (instance.gain != fastgltf::num(1)) r.emplace<ModalGain>(entity, ModalGain{instance.gain});
        }
    }

    // Build one armature per distinct armature root, consuming every skin anchored there.
    // This way a bone's pose world composes the same node transforms as the spec's global joint transform.
    std::unordered_set<uint32_t> joint_node_indices;
    struct ArmatureGroup {
        std::optional<uint32_t> ArmaNode;
        std::vector<uint32_t> SkinIndices;
    };
    std::vector<ArmatureGroup> armature_groups;
    for (uint32_t skin_index = 0; skin_index < asset.skins.size(); ++skin_index) {
        if (!used_skin[skin_index] || skin_joint_nodes[skin_index].empty()) continue;
        auto it = std::ranges::find(armature_groups, skin_arma_node[skin_index], &ArmatureGroup::ArmaNode);
        if (it == armature_groups.end()) it = armature_groups.emplace(armature_groups.end(), ArmatureGroup{skin_arma_node[skin_index], {}});
        it->SkinIndices.emplace_back(skin_index);
    }
    for (uint32_t group_index = 0; group_index < armature_groups.size(); ++group_index) {
        const auto &group = armature_groups[group_index];
        const auto arma_node = group.ArmaNode;
        if (arma_node && !traversal.InScene[*arma_node]) {
            return std::unexpected{std::format("glTF import failed for '{}': skin {} armature root node {} is not in the imported scene.", source_path.string(), group.SkinIndices.front(), *arma_node)};
        }

        // Bone nodes: every bone node on a path from a joint up to the root (exclusive), first-seen order.
        std::vector<uint32_t> source_bone_nodes;
        std::vector<bool> in_group(asset.nodes.size(), false);
        for (const auto skin_index : group.SkinIndices) {
            for (const auto joint : skin_joint_nodes[skin_index]) {
                for (std::optional<uint32_t> cur = joint; cur && cur != arma_node; cur = parents[*cur]) {
                    if (!is_bone[*cur]) continue;
                    if (in_group[*cur]) break;
                    in_group[*cur] = true;
                    source_bone_nodes.emplace_back(*cur);
                }
            }
        }

        std::unordered_map<uint32_t, std::optional<uint32_t>> bone_parent_map;
        bone_parent_map.reserve(source_bone_nodes.size());
        for (const auto node : source_bone_nodes) {
            bone_parent_map.emplace(node, FindNearestMarkedAncestor(node, parents, in_group));
        }

        auto ordered_bone_nodes = BuildParentBeforeChildJointOrder(source_bone_nodes, bone_parent_map, group_index);
        if (!ordered_bone_nodes) return std::unexpected{ordered_bone_nodes.error()};

        const auto armature_data_entity = r.create();
        auto &armature = r.emplace<Armature>(armature_data_entity);
        armature_data_entities.emplace_back(armature_data_entity);

        std::unordered_map<uint32_t, BoneId> bone_id_by_node;
        bone_id_by_node.reserve(ordered_bone_nodes->size());
        for (const auto node : *ordered_bone_nodes) {
            joint_node_indices.emplace(node);
            const auto parent_node = bone_parent_map.at(node);
            auto rest_local = ComputeJointRestLocal(group_index, node, parent_node, arma_node, parents, local_transforms);
            if (!rest_local) return std::unexpected{rest_local.error()};

            // Parents precede children in the ordered walk, so the parent's bone ID is always mapped.
            const auto parent_bone_id = parent_node ? std::optional{bone_id_by_node.at(*parent_node)} : std::nullopt;
            const auto source_name = MakeNodeName(asset, node);
            const auto bone_name = source_name.empty() ? std::format("Joint{}", node) : source_name;
            const auto bone_id = armature.AddBone(bone_name, parent_bone_id, *rest_local, node);
            bone_id_by_node.emplace(node, bone_id);
            if (const auto object_it = object_entities_by_node.find(node);
                object_it != object_entities_by_node.end() &&
                r.all_of<Instance>(object_it->second) &&
                !r.all_of<PhysicsMotion>(object_it->second) &&
                !r.all_of<BoneAttachment>(object_it->second)) {
                r.emplace<BoneAttachment>(object_it->second, armature_data_entity, bone_id);
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
        r.emplace<Transform>(armature_entity, arma_node ? ToTransform(traversal.WorldTransforms[*arma_node]) : Transform{});
        const auto skin_name = [&]() -> std::string {
            for (const auto skin_index : group.SkinIndices) {
                if (const auto &name = asset.skins[skin_index].name; !name.empty()) return std::string(name);
            }
            return {};
        }();
        EmplaceUniqueName(r, armature_entity, skin_name.empty() ? std::format("{}_Armature{}", name_prefix, group_index) : skin_name);
        if (skin_name.empty()) r.emplace<SourceEmptyName>(armature_entity);

        // Follow the root node's entity when it is an object (it may be animated), else the nearest object above it.
        if (arma_node) {
            const auto parent_node = object_entities_by_node.contains(*arma_node) ? arma_node : nearest_object_ancestor[*arma_node];
            if (const auto parent_it = parent_node ? object_entities_by_node.find(*parent_node) : object_entities_by_node.end();
                parent_it != object_entities_by_node.end()) {
                SetParentKeepWorld(r, armature_entity, parent_it->second);
            }
        }

        r.emplace<GltfObject>(armature_entity);
        if (first_armature_entity == entt::null) first_armature_entity = armature_entity;
        if (first_object_entity == entt::null) first_object_entity = armature_entity;

        for (uint32_t skin_slot = 0; skin_slot < group.SkinIndices.size(); ++skin_slot) {
            const auto skin_index = group.SkinIndices[skin_slot];
            const auto skinned_it = skinned_mesh_instances_by_skin.find(skin_index);
            if (skinned_it == skinned_mesh_instances_by_skin.end()) {
                return std::unexpected{std::format("glTF import failed '{}': skin {} is used but no mesh instances were emitted for skin binding.", source_path.string(), skin_index)};
            }
            for (const auto mesh_instance_entity : skinned_it->second) {
                if (!r.valid(mesh_instance_entity) || !r.all_of<Instance>(mesh_instance_entity)) continue;
                r.emplace_or_replace<ArmatureModifier>(mesh_instance_entity, armature_data_entity, armature_entity, skin_slot);
                // The spec ignores a skinned mesh node's own transform.
                // Identity-parent it to the armature so its world transform is the deform's space.
                r.emplace_or_replace<Transform>(mesh_instance_entity, Transform{});
                SetParent(r, mesh_instance_entity, armature_entity);
            }
        }

        // Bone instances only, their pose state is built later from the bone Transforms and rest pose.
        ::CreateBoneInstances(r, ctx.Meshes, armature_entity, armature_data_entity);
        // Mark each bone entity with its source joint NodeIndex (for SaveScene round-trip).
        const auto &bone_entities_for_source = r.get<const ArmatureObject>(armature_entity).BoneEntities;
        for (uint32_t i = 0; i < armature.Bones.size(); ++i) {
            const auto joint_node_index = armature.Bones[i].JointNodeIndex;
            if (!joint_node_index) continue;
            r.emplace<SourceNodeIndex>(bone_entities_for_source[i], *joint_node_index);
            if (*joint_node_index < asset.nodes.size() && asset.nodes[*joint_node_index].name.empty()) {
                r.emplace<SourceEmptyName>(bone_entities_for_source[i]);
            }
        }

        // Adds Child Of to bones under a physics-driven ancestor so skinned geometry follows simulation.
        // Target is the nearest ancestor object with PhysicsMotion; InverseMatrix bakes the rest offset.
        {
            const auto find_physics_ancestor_entity = [&](uint32_t node_index) -> entt::entity {
                for (std::optional<uint32_t> cur = node_index; cur;) {
                    if (const auto oit = object_entities_by_node.find(*cur);
                        oit != object_entities_by_node.end() && r.all_of<PhysicsMotion>(oit->second)) return oit->second;
                    if (*cur >= parents.size()) break;
                    cur = parents[*cur];
                }
                return entt::null;
            };
            const auto &arm_obj = r.get<const ArmatureObject>(armature_entity);
            const mat4 armature_world = ToMatrix(r.get<const WorldTransform>(armature_entity));
            for (uint32_t i = 0; i < armature.Bones.size(); ++i) {
                const auto &bone = armature.Bones[i];
                if (!bone.JointNodeIndex) continue;
                const auto target = find_physics_ancestor_entity(*bone.JointNodeIndex);
                if (target != entt::null) {
                    EnsureWorldTransform(r, target);
                    r.emplace<BoneConstraints>(
                        arm_obj.BoneEntities[i],
                        BoneConstraints{
                            .Stack = {BoneConstraint{
                                .TargetEntity = target,
                                .Influence = 1.f,
                                .Data = ChildOfData{.InverseMatrix = numeric::Inverse(ToMatrix(r.get<const WorldTransform>(target))) * (armature_world * bone.RestWorld)},
                            }}
                        }
                    );
                }
            }
        }
    }

    // Per source-derived entity: tag with source parent / sibling position / matrix-form flag.
    for (const auto [entity, sni] : r.view<const SourceNodeIndex>().each()) {
        if (sni.Value >= asset.nodes.size()) continue;
        if (const auto parent_idx = parents[sni.Value]) {
            r.emplace<SourceParentNodeIndex>(entity, *parent_idx);
            // Sibling position in parent's bounds-filtered children list.
            uint32_t sibling_idx = 0;
            for (const auto child_raw : asset.nodes[*parent_idx].children) {
                const auto child = ToIndex(child_raw, asset.nodes.size());
                if (!child) continue;
                if (*child == sni.Value) {
                    r.emplace<SourceSiblingIndex>(entity, sibling_idx);
                    break;
                }
                ++sibling_idx;
            }
        }
        if (source_matrices[sni.Value]) r.emplace<SourceMatrixTransform>(entity, *source_matrices[sni.Value]);
    }

    { // KHR_node_visibility: `visible:false` hides node *and* descendants.
        const auto hide_subtree = [&](this const auto &self, entt::entity e) -> void {
            Hide(r, e);
            for (const auto child : Children{&r, e}) self(child);
        };
        for (const auto [entity, sni] : r.view<const SourceNodeIndex>().each()) {
            if (sni.Value < asset.nodes.size() && !asset.nodes[sni.Value].visible) hide_subtree(entity);
        }
    }

    std::unordered_map<uint32_t, std::vector<std::pair<entt::entity, BoneId>>> armature_targets_by_joint_node;
    for (const auto armature_data_entity : armature_data_entities) {
        const auto &armature = r.get<const Armature>(armature_data_entity);
        for (const auto &bone : armature.Bones) {
            if (bone.JointNodeIndex) {
                armature_targets_by_joint_node[*bone.JointNodeIndex].emplace_back(armature_data_entity, bone.Id);
            }
        }
    }

    // Set up morph weight state for mesh instances with morph targets.
    // The GPU range (MorphWeightGpuRange) is allocated later.
    // Build a map: node_index -> mesh instance entity, for resolving weight animation channels.
    std::unordered_map<uint32_t, entt::entity> morph_instance_by_node;
    for (const auto &object : source_objects) {
        if (object.ObjectType != gltf::Object::Type::Mesh || !object.MeshIndex) continue;
        if (*object.MeshIndex >= mesh_morphs.size()) continue;
        const auto obj_it = object_entities_by_node.find(object.NodeIndex);
        if (obj_it == object_entities_by_node.end()) continue;
        const auto instance_entity = obj_it->second;
        if (!r.all_of<Instance>(instance_entity)) continue;

        const auto &morph = mesh_morphs[*object.MeshIndex];
        if (morph.TargetCount == 0) continue;

        auto weights = [&] {
            if (!object.NodeWeights) return morph.DefaultWeights;
            std::vector<float> w(morph.TargetCount, 0.f);
            std::copy_n(object.NodeWeights->begin(), std::min(uint32_t(object.NodeWeights->size()), morph.TargetCount), w.begin());
            return w;
        }();
        r.emplace<MorphWeightState>(instance_entity, MorphWeightState{.Weights = std::move(weights)});
        morph_instance_by_node[object.NodeIndex] = instance_entity;
    }

    // Resolve object/node transform animations (empties, meshes, cameras, lights).
    // Channels targeting skin joints are handled by ArmatureAnimation and skipped here.
    std::unordered_map<entt::entity, Transform> node_anim_bindings;
    node_anim_bindings.reserve(object_entities_by_node.size());
    for (const auto &[node_index, object_entity] : object_entities_by_node) {
        if (r.valid(object_entity) && node_index < local_transforms.size()) {
            node_anim_bindings.emplace(object_entity, local_transforms[node_index]);
        }
    }

    bool imported_animation = false;
    const auto append_node_clip = [&](entt::entity object_entity, ::AnimationClip &&resolved_clip) {
        if (resolved_clip.Channels.empty()) return;
        imported_animation = true;
        if (auto *existing = r.try_get<NodeTransformAnimation>(object_entity)) {
            existing->Clips.emplace_back(std::move(resolved_clip));
            return;
        }
        if (!node_anim_bindings.contains(object_entity)) return; // needs a known local transform
        r.emplace<NodeTransformAnimation>(
            object_entity,
            NodeTransformAnimation{.Clips = {std::move(resolved_clip)}, .ActiveClipIndex = 0}
        );
    };

    // Parse source channels directly into target ECS clips in one pass.
    // AnimationOrder records source names for animations with at least one valid channel.
    std::vector<std::string> animation_order;
    animation_order.reserve(asset.animations.size());
    struct ChannelTargetSpec {
        AnimationPath Path;
        size_t ComponentCount;
    };
    for (const auto &anim : asset.animations) {
        std::unordered_map<entt::entity, ::AnimationClip> armature_clips_by_entity;
        std::unordered_map<entt::entity, MorphWeightClip> morph_clips_by_entity;
        std::unordered_map<entt::entity, ::AnimationClip> node_clips_by_entity;
        const std::string anim_name(anim.name);
        float max_time = 0;
        bool any_channel = false;

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

            const auto interp = ToInterp(sampler.interpolation);

            std::vector<float> times(input_accessor.count);
            fastgltf::copyFromAccessor<float>(asset, input_accessor, times.data());

            std::vector<float> values;
            if (target_spec->Path == AnimationPath::Weights) {
                values.resize(output_accessor.count);
                fastgltf::copyFromAccessor<float>(asset, output_accessor, values.data());
            } else {
                values.resize(output_accessor.count * target_spec->ComponentCount);
                if (target_spec->ComponentCount == 4) fastgltf::copyFromAccessor<vec4>(asset, output_accessor, reinterpret_cast<vec4 *>(values.data()));
                else fastgltf::copyFromAccessor<vec3>(asset, output_accessor, reinterpret_cast<vec3 *>(values.data()));
            }

            if (!times.empty()) max_time = std::max(max_time, times.back());
            any_channel = true;
            const uint32_t target_node_index = *channel.nodeIndex;

            if (target_spec->Path == AnimationPath::Weights) {
                const auto inst_it = morph_instance_by_node.find(target_node_index);
                if (inst_it == morph_instance_by_node.end()) continue;
                auto &resolved_clip = morph_clips_by_entity
                                          .try_emplace(inst_it->second, MorphWeightClip{.Name = anim_name, .DurationSeconds = 0.f, .Channels = {}})
                                          .first->second;
                resolved_clip.Channels.emplace_back(MorphWeightChannel{.Interp = interp, .TimesSeconds = std::move(times), .Values = std::move(values)});
                continue;
            }

            if (const auto armature_it = armature_targets_by_joint_node.find(target_node_index);
                armature_it != armature_targets_by_joint_node.end()) {
                for (const auto &[target_data_entity, bone_id] : armature_it->second) {
                    const auto &armature = r.get<const Armature>(target_data_entity);
                    const auto bone_index = armature.FindBoneIndex(bone_id).value_or(InvalidBoneIndex);
                    auto &resolved_clip = armature_clips_by_entity
                                              .try_emplace(target_data_entity, ::AnimationClip{.Name = anim_name, .DurationSeconds = 0.f, .Channels = {}})
                                              .first->second;
                    resolved_clip.Channels.emplace_back(::AnimationChannel{.BoneIndex = bone_index, .TargetBoneId = bone_id, .Target = target_spec->Path, .Interp = interp, .TimesSeconds = times, .Values = values});
                }
                continue;
            }

            if (joint_node_indices.contains(target_node_index)) continue;

            const auto object_it = object_entities_by_node.find(target_node_index);
            if (object_it != object_entities_by_node.end() && r.valid(object_it->second)) {
                auto &resolved_clip = node_clips_by_entity
                                          .try_emplace(object_it->second, ::AnimationClip{.Name = anim_name, .DurationSeconds = 0.f, .Channels = {}})
                                          .first->second;
                resolved_clip.Channels.emplace_back(::AnimationChannel{.BoneIndex = 0, .Target = target_spec->Path, .Interp = interp, .TimesSeconds = std::move(times), .Values = std::move(values)});
            }
        }

        if (!any_channel) continue;
        animation_order.emplace_back(std::move(anim_name));

        for (auto &[_, c] : armature_clips_by_entity) c.DurationSeconds = max_time;
        for (auto &[_, c] : morph_clips_by_entity) c.DurationSeconds = max_time;
        for (auto &[_, c] : node_clips_by_entity) c.DurationSeconds = max_time;

        for (auto &[target_data_entity, resolved_clip] : armature_clips_by_entity) imported_animation |= AppendClip<ArmatureAnimation>(r, target_data_entity, std::move(resolved_clip));
        for (auto &[instance_entity, resolved_clip] : morph_clips_by_entity) imported_animation |= AppendClip<MorphWeightAnimation>(r, instance_entity, std::move(resolved_clip));
        for (auto &[object_entity, resolved_clip] : node_clips_by_entity) append_node_clip(object_entity, std::move(resolved_clip));
    }
    r.patch<gltf::SourceAssets>(viewport, [&](auto &a) { a.AnimationOrder = std::move(animation_order); });

    { // Get timeline range from imported animation durations
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
        if (max_dur > 0) r.patch<TimelineRange>(viewport, [&](auto &r) { r.EndFrame = int(std::ceil(max_dur * r.Fps)); });
    }

    if (source_ibl) {
        if (auto *prev = r.try_get<PendingEnvironmentImport>(viewport)) prev_pending_env_backup = *prev;
        const auto [diffuse_slot, specular_slot] = AllocateIblCubeSlots(ctx.Slots);
        r.emplace_or_replace<PendingEnvironmentImport>(viewport, *source_ibl, diffuse_slot, specular_slot);
        r.remove<PendingSceneWorldClear>(viewport);
        replaced_pending_env = true;
    } else {
        r.emplace_or_replace<PendingSceneWorldClear>(viewport);
    }
    // Import-time UX default: show an imported world, hide the (empty) default world.
    // Kept out of the reactive world passes so a snapshot restore reproduces the saved WorldOpacity rather than re-forcing this.
    if (r.all_of<RenderedLighting>(viewport)) r.patch<RenderedLighting>(viewport, [&](auto &l) { l.WorldOpacity = source_ibl ? 1.f : 0.f; });

    // First-class scene entities, one per source scene. The default scene is the active one.
    std::vector<entt::entity> scene_entities;
    scene_entities.reserve(asset.scenes.size());
    for (uint32_t i = 0; i < asset.scenes.size(); ++i) {
        const auto se = r.create();
        r.emplace<Scene>(se, std::string{asset.scenes[i].name});
        r.emplace<SourceSceneIndex>(se, i);
        if (i == scene_index) r.emplace<ActiveScene>(se);
        scene_entities.emplace_back(se);
    }
    // Multi-scene only: record each node's scene membership as references to those scene entities.
    if (asset.scenes.size() > 1) {
        for (const auto [e, sni] : r.view<const SourceNodeIndex>().each()) {
            if (sni.Value >= node_to_scene_mask.size()) continue;
            const auto mask = node_to_scene_mask[sni.Value];
            std::vector<entt::entity> scenes;
            for (uint32_t i = 0; i < scene_entities.size(); ++i) {
                if (mask & (1u << i)) scenes.emplace_back(scene_entities[i]);
            }
            if (!scenes.empty()) r.emplace<SceneMembership>(e, std::move(scenes));
        }
    }
    ApplySceneVisibility(r);
    ApplyActiveSceneSelection(r);
    if (!materialized_textures.empty()) {
        auto &manifest = r.get_or_emplace<MaterializedTextures>(viewport);
        manifest.Items.insert(manifest.Items.end(), std::make_move_iterator(materialized_textures.begin()), std::make_move_iterator(materialized_textures.end()));
    }
    import_rollback_guard.Enabled = false;

    return gltf::LoadResult{.FirstCameraObject = first_camera_object_entity, .ImportedAnimation = imported_animation};
}

void SwitchActiveScene(entt::registry &r, entt::entity scene) {
    if (!r.all_of<Scene>(scene) || r.all_of<ActiveScene>(scene)) return;
    r.clear<ActiveScene>();
    r.emplace<ActiveScene>(scene);
    ApplySceneVisibility(r);
    ApplyActiveSceneSelection(r);
}

static_assert(uint32_t(ExtrasCategory::Images) == uint32_t(fastgltf::Category::Images));
static_assert(uint32_t(ExtrasCategory::Samplers) == uint32_t(fastgltf::Category::Samplers));
static_assert(uint32_t(ExtrasCategory::Textures) == uint32_t(fastgltf::Category::Textures));
static_assert(uint32_t(ExtrasCategory::Animations) == uint32_t(fastgltf::Category::Animations));
static_assert(uint32_t(ExtrasCategory::Cameras) == uint32_t(fastgltf::Category::Cameras));
static_assert(uint32_t(ExtrasCategory::Materials) == uint32_t(fastgltf::Category::Materials));
static_assert(uint32_t(ExtrasCategory::Meshes) == uint32_t(fastgltf::Category::Meshes));
static_assert(uint32_t(ExtrasCategory::Skins) == uint32_t(fastgltf::Category::Skins));
static_assert(uint32_t(ExtrasCategory::Nodes) == uint32_t(fastgltf::Category::Nodes));
static_assert(uint32_t(ExtrasCategory::Scenes) == uint32_t(fastgltf::Category::Scenes));
static_assert(uint32_t(ExtrasCategory::Lights) == uint32_t(fastgltf::Category::Lights));
static_assert(uint32_t(ExtrasCategory::ImageBasedLights) == uint32_t(fastgltf::Category::ImageBasedLights));

std::optional<std::string_view> GetExtras(const SourceAssets &sa, ExtrasCategory cat, uint32_t source_index) {
    const auto key = (uint64_t(uint32_t(cat)) << 32) | uint64_t(source_index);
    if (const auto it = sa.ExtrasByEntity.find(key); it != sa.ExtrasByEntity.end()) return std::string_view{it->second};
    return std::nullopt;
}
} // namespace gltf
