#include "GltfScene.h"

#include "../ImageEncode.h"
#include "AnimationData.h"
#include "AnimationTimeline.h"
#include "Armature.h"
#include "Camera.h"
#include "Instance.h"
#include "NodeTransformAnimation.h"
#include "Path.h"
#include "PbrFeature.h"
#include "SceneOps.h"
#include "SceneTextures.h"
#include "SceneTree.h"
#include "Timer.h"
#include "TransformMath.h"
#include "Variant.h"
#include "mesh/MeshStore.h"
#include "mesh/MorphTargetData.h"
#include "physics/PhysicsTypes.h"
#include "scene_impl/SceneBuffers.h"
#include "scene_impl/SceneComponents.h"
#include "vulkan/Slots.h"

#include <fastgltf/base64.hpp>
#include <fastgltf/core.hpp>
#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/types.hpp>
#include <simdjson.h>

#include <entt/entity/registry.hpp>

#include <algorithm>
#include <cstring>
#include <format>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <numbers>
#include <numeric>
#include <span>
#include <unordered_map>
#include <unordered_set>

// Load-only intermediate carrier (TU-local). Holds parsed-but-not-yet-uploaded geometry: load
// builds it from fastgltf accessors, runs `MeshStore::PlanCreate` for the whole batch (so arena
// reserves happen before any CreateMesh), and then drains into ECS. PlanCreate's batch-then-
// commit contract is what blocks a single-pass collapse — the bulk vertex/face data legitimately
// outlives the parse. Save reads MeshStore vertices directly per-primitive, no carrier needed.
namespace gltf {
struct MeshData {
    std::optional<::MeshData> Triangles, Lines, Points;
    ::MeshVertexAttributes TriangleAttrs, LineAttrs, PointAttrs;
    ::MeshPrimitives TrianglePrimitives;
    std::optional<ArmatureDeformData> DeformData;
    std::optional<MorphTargetData> MorphData;
    std::string Name;
};

// Per-node KHR_physics_rigid_bodies data, converted from fastgltf to engine types up-front so
// the physics consumer pass below can drain into ECS without re-doing variant visits per node.
// All other per-node state is read directly from `asset.nodes[i]` / `parents` / `traversal` /
// `local_transforms` / `source_matrices` / `is_joint` at the consumer site (no caching layer).
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

} // namespace gltf

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

    std::vector<std::byte> bytes(end);
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char *>(bytes.data()), std::streamsize(end));
    if (!file) return std::unexpected{std::format("Failed to read image file '{}'.", path.string())};
    return bytes;
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
                auto bytes = ReadFileBytes(image_path);
                if (!bytes) return std::unexpected{std::move(bytes.error())};
                // Hold bytes only until upload — Scene::ProcessComponentEvents clears them after
                // materialization since SourceAbsPath is the persistence for external URIs.
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
            for (uint32_t i = 0; i < position_accessor.count; ++i) (*attrs.Colors0)[base_vertex + i] = vec4{colors[i], 1.f};
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
            for (uint32_t i = 0; i < position_accessor.count; ++i) (*attrs.Colors0)[base_vertex + i] = vec4{colors[i], 1.f};
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

// Always emplaces a MeshData so meshes stays index-aligned with asset.meshes; callers
// check Triangles/Lines/Points presence before referencing the mesh.
std::expected<uint32_t, std::string> EnsureMeshData(const fastgltf::Asset &asset, uint32_t source_mesh_index, std::vector<MeshData> &meshes, std::unordered_map<uint32_t, uint32_t> &mesh_index_map, std::size_t material_count) {
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
    std::vector<uint32_t> primitive_material_indices(source_mesh.primitives.size(), material_count == 0 ? 0u : uint32_t(material_count - 1u));
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
        if (primitive.type == fastgltf::PrimitiveType::Points) {
            AppendNonTrianglePrimitive(asset, primitive, points, point_attrs);
            continue;
        }
        if (primitive.type == fastgltf::PrimitiveType::Lines || primitive.type == fastgltf::PrimitiveType::LineStrip || primitive.type == fastgltf::PrimitiveType::LineLoop) {
            AppendNonTrianglePrimitive(asset, primitive, lines, line_attrs);
            continue;
        }
        const uint32_t prev_vertex_count = mesh.Positions.size(), prev_face_count = mesh.Faces.size();
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
            uint32_t src_off{0}, dst_vert_off{0};
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

    const auto mesh_index = meshes.size();
    meshes.emplace_back(MeshData{
        .Triangles = has_triangles ? std::optional{std::move(mesh)} : std::nullopt,
        .Lines = has_lines ? std::optional{std::move(lines)} : std::nullopt,
        .Points = has_points ? std::optional{std::move(points)} : std::nullopt,
        .TriangleAttrs = std::move(mesh_attrs),
        .LineAttrs = std::move(line_attrs),
        .PointAttrs = std::move(point_attrs),
        .TrianglePrimitives = has_triangles ? ::MeshPrimitives{std::move(face_primitive_indices), std::move(primitive_material_indices), std::move(vertex_counts), std::move(attribute_flags), std::move(has_source_indices), std::move(variant_mappings)} : ::MeshPrimitives{},
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
} // namespace

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
fastgltf::Optional<T> ToFgOpt(const std::optional<U> &o) { return o ? fastgltf::Optional<T>{T(*o)} : fastgltf::Optional<T>{}; }
template<typename T, typename U, typename Fn>
fastgltf::Optional<T> ToFgOpt(const std::optional<U> &o, Fn &&fn) { return o ? fastgltf::Optional<T>{fn(*o)} : fastgltf::Optional<T>{}; }

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

// Strided variant: copy field `field` of each element in `data` into the binary blob (4-byte
// aligned), no intermediate vector materialized. Returns the starting byte offset.
template<typename T, typename V>
uint32_t AppendField(std::vector<std::byte> &buffer, std::span<const V> data, T V::*field) {
    const uint32_t offset = buffer.size();
    buffer.resize(offset + data.size() * sizeof(T));
    auto *out = reinterpret_cast<T *>(buffer.data() + offset);
    for (std::size_t i = 0; i < data.size(); ++i) out[i] = data[i].*field;
    while (buffer.size() % 4 != 0) buffer.emplace_back(std::byte{0});
    return offset;
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

fastgltf::Camera ConvertCameraToFg(const ::Camera &cam, std::string_view name) {
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
        cam
    );
    return fastgltf::Camera{.camera = std::move(camera), .name = ToFgStr(name)};
}

std::optional<ImageBasedLight> ConvertIBL(const fastgltf::Asset &asset, std::size_t scene_index) {
    const auto ibl_idx = ToIndex(asset.scenes[scene_index].imageBasedLightIndex, asset.imageBasedLights.size());
    if (!ibl_idx) return std::nullopt;
    const auto &src_ibl = asset.imageBasedLights[*ibl_idx];
    ImageBasedLight ibl{
        .Rotation = glm::normalize(quat{src_ibl.rotation[3], src_ibl.rotation[0], src_ibl.rotation[1], src_ibl.rotation[2]}),
        .SpecularImageSize = src_ibl.specularImageSize,
        .Intensity = std::max(0.f, src_ibl.intensity),
        .Name = src_ibl.name.empty() ? std::format("ImageBasedLight{}", *ibl_idx) : std::string{src_ibl.name},
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
    return ibl;
}

fastgltf::Light ConvertLightToFg(const PunctualLight &pl, std::string_view name) {
    const auto type = pl.Type == PunctualLightType::Point ? fastgltf::LightType::Point : pl.Type == PunctualLightType::Spot ? fastgltf::LightType::Spot :
                                                                                                                              fastgltf::LightType::Directional;
    const bool is_spot = type == fastgltf::LightType::Spot;
    return fastgltf::Light{
        .type = type,
        .color = {pl.Color.x, pl.Color.y, pl.Color.z},
        .intensity = pl.Intensity,
        .range = (type != fastgltf::LightType::Directional && pl.Range > 0) ? fastgltf::Optional<fastgltf::num>{pl.Range} : fastgltf::Optional<fastgltf::num>{},
        .innerConeAngle = is_spot ? fastgltf::Optional<fastgltf::num>{std::acos(std::clamp(pl.InnerConeCos, -1.f, 1.f))} : fastgltf::Optional<fastgltf::num>{},
        .outerConeAngle = is_spot ? fastgltf::Optional<fastgltf::num>{std::acos(std::clamp(pl.OuterConeCos, -1.f, 1.f))} : fastgltf::Optional<fastgltf::num>{},
        .name = ToFgStr(name),
    };
}

} // namespace

std::expected<PopulateResult, std::string> LoadGltf(const std::filesystem::path &source_path, PopulateContext ctx) {
    const Timer timer{"LoadGltf"};

    ExtrasMap extras;
    auto parsed_asset = ParseAsset(source_path, &extras);
    if (!parsed_asset) return std::unexpected{parsed_asset.error()};

    auto &asset = *parsed_asset;
    if (asset.scenes.empty()) return std::unexpected{std::format("glTF '{}' has no scenes.", source_path.string())};

    const auto scene_index = asset.defaultScene.value_or(0);
    if (scene_index >= asset.scenes.size()) return std::unexpected{std::format("glTF '{}' has invalid default scene index.", source_path.string())};

    // Build SourceAssets in-place, no local intermediate vectors. The struct lives on
    // SceneEntity for the rest of the load (and beyond — needed for save round-trip), and the
    // resolve_texture_slot lambda below reads samplers/images/textures via this stored ref.
    gltf::SourceAssets source_assets{
        .Copyright = asset.assetInfo ? std::string{asset.assetInfo->copyright} : std::string{},
        .Generator = asset.assetInfo ? std::string{asset.assetInfo->generator} : std::string{},
        .MinVersion = asset.assetInfo ? std::string{asset.assetInfo->minVersion} : std::string{},
        .AssetExtras = asset.assetInfo ? std::string{asset.assetInfo->extras} : std::string{},
        .AssetExtensions = asset.assetInfo ? std::string{asset.assetInfo->extensions} : std::string{},
        .DefaultSceneName = std::string{asset.scenes[scene_index].name},
        .DefaultSceneRoots = {},
        .ExtensionsRequired = {},
        .MaterialVariants = {},
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
    for (uint32_t image_index = 0; image_index < asset.images.size(); ++image_index) {
        auto image_result = ReadImage(asset, image_index, source_path.parent_path());
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

    // Per source material: PBRMaterial (texture slots are gltf indices here; remapped to bindless
    // slots in the emit loop below) + the lossy delta `MaterialSourceMeta` carries for round-trip.
    // Trailing entry is the synthetic "DefaultMaterial" fallback (see below).
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

        meta.TextureSlots = {
            pbr.BaseColorTexture.Slot,
            pbr.MetallicRoughnessTexture.Slot,
            pbr.NormalTexture.Slot,
            pbr.OcclusionTexture.Slot,
            pbr.EmissiveTexture.Slot,
            pbr.Specular.Texture.Slot,
            pbr.Specular.ColorTexture.Slot,
            pbr.Sheen.ColorTexture.Slot,
            pbr.Sheen.RoughnessTexture.Slot,
            pbr.Transmission.Texture.Slot,
            pbr.DiffuseTransmission.Texture.Slot,
            pbr.DiffuseTransmission.ColorTexture.Slot,
            pbr.Volume.ThicknessTexture.Slot,
            pbr.Clearcoat.Texture.Slot,
            pbr.Clearcoat.RoughnessTexture.Slot,
            pbr.Clearcoat.NormalTexture.Slot,
            pbr.Anisotropy.Texture.Slot,
            pbr.Iridescence.Texture.Slot,
            pbr.Iridescence.ThicknessTexture.Slot,
        };
        material_metas.emplace_back(std::move(meta));
        source_materials.emplace_back(std::move(pbr));
    }
    // Synthetic fallback used as the default material when a primitive references no material.
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

    // Parse KHR_physics_rigid_bodies document-level resources. Create entt entities directly —
    // no intermediate staging vectors. Collision filters resolve names later (need the dedupe map),
    // so they're emitted in the consumer block below.
    std::vector<entt::entity> physics_material_entities, physics_jointdef_entities;
    {
        const auto ToCombineMode = [](fastgltf::CombineMode m) {
            switch (m) {
                case fastgltf::CombineMode::Minimum: return PhysicsCombineMode::Minimum;
                case fastgltf::CombineMode::Maximum: return PhysicsCombineMode::Maximum;
                case fastgltf::CombineMode::Multiply: return PhysicsCombineMode::Multiply;
                default: return PhysicsCombineMode::Average;
            }
        };
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

    // Convert implicit shapes for physics geometry references.
    // Mesh-backed shapes leave MeshEntity as null_entity here - SceneGltf.cpp resolves it via node-to-entity mapping.
    const auto ToPhysicsShape = [&](const fastgltf::Geometry &geom) -> PhysicsShape {
        if (geom.shape && *geom.shape < asset.shapes.size()) {
            return std::visit(
                overloaded{
                    [](const fastgltf::BoxShape &s) -> PhysicsShape { return physics::Box{ToVec3(s.size)}; },
                    [](const fastgltf::SphereShape &s) -> PhysicsShape { return physics::Sphere{s.radius}; },
                    [](const fastgltf::CapsuleShape &s) -> PhysicsShape { return physics::Capsule{s.height, s.radiusTop, s.radiusBottom}; },
                    [](const fastgltf::CylinderShape &s) -> PhysicsShape { return physics::Cylinder{s.height, s.radiusTop, s.radiusBottom}; },
                    [](const fastgltf::PlaneShape &s) -> PhysicsShape { return physics::Plane{s.sizeX, s.sizeZ, s.doubleSided}; },
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

    // Load every source mesh in index order, so source_meshes[i] aligns with asset.meshes[i].
    // This preserves node.meshIndex values on round-trip. Empty-geometry meshes still get a
    // slot (Triangles/Lines/Points all nullopt), and node traversal below checks for geometry
    // presence before pointing a Node at one.
    std::unordered_map<uint32_t, uint32_t> mesh_index_map;
    std::vector<MeshData> source_meshes;
    source_meshes.reserve(asset.meshes.size());
    for (uint32_t source_mesh_index = 0; source_mesh_index < asset.meshes.size(); ++source_mesh_index) {
        auto ensured = EnsureMeshData(asset, source_mesh_index, source_meshes, mesh_index_map, source_materials.size());
        if (!ensured) return std::unexpected{std::move(ensured.error())};
    }

    // Per-node physics conversion (only for nodes carrying KHR_physics_rigid_bodies). All other
    // per-node info — local/world transform, parents/children, name, mesh/skin/camera/light refs —
    // is read at the consumer site directly from `asset.nodes[node_index]` / `parents` / etc.
    std::vector<NodePhysics> source_node_physics(asset.nodes.size());
    for (uint32_t node_index = 0; node_index < asset.nodes.size(); ++node_index) {
        const auto &source_node = asset.nodes[node_index];
        if (const auto &rb = source_node.physicsRigidBody) {
            auto &node = source_node_physics[node_index];
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
                                if (n < asset.nodes.size()) trigger.NodeIndices.emplace_back(uint32_t(n));
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
        if (!traversal.InScene[node_index]) continue;
        // Joint nodes are bone-only unless they also carry renderable mesh data.
        const bool has_mesh = ToIndex(asset.nodes[node_index].meshIndex, asset.meshes.size()).has_value();
        is_object_emitted[node_index] = has_mesh || !is_joint[node_index];
    }

    std::vector<std::optional<uint32_t>> nearest_object_ancestor(asset.nodes.size());
    for (uint32_t node_index = 0; node_index < asset.nodes.size(); ++node_index) {
        nearest_object_ancestor[node_index] = FindNearestEmittedObjectAncestor(node_index, parents, is_object_emitted);
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

    // Quick scan: any skin with at least one valid joint reference and used_skin[i] is usable
    // by the merged build/consume loop below. Used only for the early-exit guard.
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

    auto &R = ctx.R;
    const auto SceneEntity = ctx.SceneEntity;
    auto &texture_store = ctx.Textures;
    const auto texture_start = texture_store.Textures.size();
    const auto material_start = ctx.Buffers.Materials.Count();
    const auto material_name_start = R.get<const MaterialStore>(SceneEntity).Names.size();
    const auto pending_texture_start = R.all_of<PendingTextureUploads>(SceneEntity) ? R.get<const PendingTextureUploads>(SceneEntity).Items.size() : size_t{0};
    bool replaced_pending_env = false;
    std::optional<PendingEnvironmentImport> prev_pending_env_backup;
    const auto rollback_import_side_effects = [&] {
        if (texture_store.Textures.size() > texture_start) {
            ReleaseSamplerSlots(ctx.Slots, CollectSamplerSlots(std::span<const TextureEntry>{texture_store.Textures}.subspan(texture_start)));
            texture_store.Textures.resize(texture_start);
        }
        if (auto *pending = R.try_get<PendingTextureUploads>(SceneEntity); pending && pending->Items.size() > pending_texture_start) {
            for (size_t i = pending_texture_start; i < pending->Items.size(); ++i) {
                ReleaseSamplerSlots(ctx.Slots, std::span{&pending->Items[i].SamplerSlot, 1});
            }
            pending->Items.resize(pending_texture_start);
            if (pending->Items.empty()) R.remove<PendingTextureUploads>(SceneEntity);
        }
        if (replaced_pending_env) {
            if (auto *cur = R.try_get<PendingEnvironmentImport>(SceneEntity)) {
                ReleaseCubeSamplerSlot(ctx.Slots, cur->DiffuseCubeSlot);
                ReleaseCubeSamplerSlot(ctx.Slots, cur->SpecularCubeSlot);
            }
            if (prev_pending_env_backup) R.emplace_or_replace<PendingEnvironmentImport>(SceneEntity, std::move(*prev_pending_env_backup));
            else R.remove<PendingEnvironmentImport>(SceneEntity);
        }
        if (ctx.Buffers.Materials.Count() > material_start) ctx.Buffers.Materials.SetCount(material_start);
        R.patch<MaterialStore>(
            SceneEntity,
            [&](auto &store) {
                if (store.Names.size() > material_name_start) store.Names.resize(material_name_start);
            }
        );
    };
    struct ImportRollbackGuard {
        decltype(rollback_import_side_effects) &Rollback;
        bool Enabled{true};
        ~ImportRollbackGuard() {
            if (Enabled) Rollback();
        }
    };
    ImportRollbackGuard import_rollback_guard{rollback_import_side_effects};

    const auto resolve_image_index = [&](const gltf::Texture &texture) -> std::optional<uint32_t> {
        if (texture.ImageIndex) return texture.ImageIndex;
        if (texture.WebpImageIndex) return texture.WebpImageIndex;
        if (texture.BasisuImageIndex) return texture.BasisuImageIndex;
        return texture.DdsImageIndex;
    };

    // Finalize SourceAssets: scene roots, extensions/variants lists, IBL, MaterialMetas (built
    // alongside source_materials above). Then emplace and bind a const ref for downstream lookups.
    auto source_ibl = ConvertIBL(asset, scene_index);
    source_assets.ImageBasedLight = source_ibl;
    source_assets.DefaultSceneRoots.reserve(asset.scenes[scene_index].nodeIndices.size());
    for (const auto n : asset.scenes[scene_index].nodeIndices) {
        if (const auto idx = ToIndex(n, asset.nodes.size())) source_assets.DefaultSceneRoots.emplace_back(*idx);
    }
    source_assets.ExtensionsRequired.reserve(asset.extensionsRequired.size());
    for (const auto &e : asset.extensionsRequired) source_assets.ExtensionsRequired.emplace_back(e);
    source_assets.MaterialVariants.reserve(asset.materialVariants.size());
    for (const auto &v : asset.materialVariants) source_assets.MaterialVariants.emplace_back(v);
    source_assets.MaterialMetas = std::move(material_metas);
    const auto &sa = R.emplace_or_replace<gltf::SourceAssets>(SceneEntity, std::move(source_assets));

    std::vector<PendingTextureUpload> new_pending_textures;
    std::unordered_map<uint64_t, uint32_t> texture_slot_cache;
    // Cache on the resolved (image_index, sampler_index, color_space) rather than glTF texture index,
    // so that multiple glTF textures referencing the same image+sampler share a single TextureEntry and sampler slot.
    const auto texture_cache_key = [](uint32_t image_index, uint32_t sampler_index, TextureColorSpace color_space) {
        return (uint64_t(image_index) << 33u) | (uint64_t(sampler_index) << 1u) | (color_space == TextureColorSpace::Srgb ? 1u : 0u);
    };
    const auto resolve_texture_slot = [&](uint32_t texture_index, TextureColorSpace color_space) -> std::expected<uint32_t, std::string> {
        if (texture_index >= sa.Textures.size()) return InvalidSlot;

        const auto &src_texture = sa.Textures[texture_index];
        const auto image_index = resolve_image_index(src_texture);
        if (!image_index || *image_index >= sa.Images.size()) return InvalidSlot;

        const auto sampler_index = src_texture.SamplerIndex.value_or(InvalidSlot);
        const auto cache_key = texture_cache_key(*image_index, sampler_index, color_space);
        if (const auto it = texture_slot_cache.find(cache_key); it != texture_slot_cache.end()) return it->second;

        const auto *src_sampler = src_texture.SamplerIndex && *src_texture.SamplerIndex < sa.Samplers.size() ?
            &sa.Samplers[*src_texture.SamplerIndex] :
            nullptr;
        static constexpr auto ToSamplerAddressMode = [](gltf::Wrap wrap) {
            switch (wrap) {
                case gltf::Wrap::ClampToEdge: return vk::SamplerAddressMode::eClampToEdge;
                case gltf::Wrap::MirroredRepeat: return vk::SamplerAddressMode::eMirroredRepeat;
                case gltf::Wrap::Repeat: return vk::SamplerAddressMode::eRepeat;
            }
            return vk::SamplerAddressMode::eRepeat;
        };
        static constexpr auto ToSamplerConfig = [](const gltf::Sampler *sampler) -> SamplerConfig {
            if (!sampler) return {.MinFilter = vk::Filter::eLinear, .MagFilter = vk::Filter::eLinear, .MipmapMode = vk::SamplerMipmapMode::eLinear, .UsesMipmaps = true};

            const auto mag_filter = sampler->MagFilter && *sampler->MagFilter == gltf::Filter::Nearest ? vk::Filter::eNearest : vk::Filter::eLinear;
            switch (sampler->MinFilter.value_or(gltf::Filter::LinearMipMapLinear)) {
                case gltf::Filter::Nearest:
                    return {.MinFilter = vk::Filter::eNearest, .MagFilter = mag_filter, .MipmapMode = vk::SamplerMipmapMode::eNearest, .UsesMipmaps = false};
                case gltf::Filter::Linear:
                    return {.MinFilter = vk::Filter::eLinear, .MagFilter = mag_filter, .MipmapMode = vk::SamplerMipmapMode::eNearest, .UsesMipmaps = false};
                case gltf::Filter::NearestMipMapNearest:
                    return {.MinFilter = vk::Filter::eNearest, .MagFilter = mag_filter, .MipmapMode = vk::SamplerMipmapMode::eNearest, .UsesMipmaps = true};
                case gltf::Filter::LinearMipMapNearest:
                    return {.MinFilter = vk::Filter::eLinear, .MagFilter = mag_filter, .MipmapMode = vk::SamplerMipmapMode::eNearest, .UsesMipmaps = true};
                case gltf::Filter::NearestMipMapLinear:
                    return {.MinFilter = vk::Filter::eNearest, .MagFilter = mag_filter, .MipmapMode = vk::SamplerMipmapMode::eLinear, .UsesMipmaps = true};
                case gltf::Filter::LinearMipMapLinear:
                    return {.MinFilter = vk::Filter::eLinear, .MagFilter = mag_filter, .MipmapMode = vk::SamplerMipmapMode::eLinear, .UsesMipmaps = true};
            }
            return {.MinFilter = vk::Filter::eLinear, .MagFilter = mag_filter, .MipmapMode = vk::SamplerMipmapMode::eLinear, .UsesMipmaps = true};
        };

        const auto sampler_config = ToSamplerConfig(src_sampler);
        const auto wrap_s = src_sampler ? ToSamplerAddressMode(src_sampler->WrapS) : vk::SamplerAddressMode::eRepeat;
        const auto wrap_t = src_sampler ? ToSamplerAddressMode(src_sampler->WrapT) : vk::SamplerAddressMode::eRepeat;
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
    ctx.Buffers.Materials.Reserve(material_count + source_materials.size());
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
        // Resolves a texture slot in-place: tex.Slot starts as a glTF texture index (stored by GltfLoader),
        // and is replaced with the bindless sampler slot. UV fields are already correct from the loader.
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
        if (auto result = [&]() -> std::expected<void, std::string> {
                std::expected<void, std::string> resolve_result{};
                const auto check = [&](TextureInfo &tex, TextureColorSpace color_space, std::string_view texture_label) {
                    resolve_result = resolve_texture(tex, color_space, texture_label);
                    return resolve_result.has_value();
                };
                if (
                    !check(gpu_material.BaseColorTexture, TextureColorSpace::Srgb, "baseColor") ||
                    !check(gpu_material.MetallicRoughnessTexture, TextureColorSpace::Linear, "metallicRoughness") ||
                    !check(gpu_material.NormalTexture, TextureColorSpace::Linear, "normal") ||
                    !check(gpu_material.OcclusionTexture, TextureColorSpace::Linear, "occlusion") ||
                    !check(gpu_material.EmissiveTexture, TextureColorSpace::Srgb, "emissive") ||
                    !check(gpu_material.Specular.Texture, TextureColorSpace::Linear, "specular") ||
                    !check(gpu_material.Specular.ColorTexture, TextureColorSpace::Srgb, "specularColor") ||
                    !check(gpu_material.Sheen.ColorTexture, TextureColorSpace::Srgb, "sheenColor") ||
                    !check(gpu_material.Sheen.RoughnessTexture, TextureColorSpace::Linear, "sheenRoughness") ||
                    !check(gpu_material.Transmission.Texture, TextureColorSpace::Linear, "transmission") ||
                    !check(gpu_material.DiffuseTransmission.Texture, TextureColorSpace::Linear, "diffuseTransmission") ||
                    !check(gpu_material.DiffuseTransmission.ColorTexture, TextureColorSpace::Srgb, "diffuseTransmissionColor") ||
                    !check(gpu_material.Volume.ThicknessTexture, TextureColorSpace::Linear, "thickness") ||
                    !check(gpu_material.Clearcoat.Texture, TextureColorSpace::Linear, "clearcoat") ||
                    !check(gpu_material.Clearcoat.RoughnessTexture, TextureColorSpace::Linear, "clearcoatRoughness") ||
                    !check(gpu_material.Clearcoat.NormalTexture, TextureColorSpace::Linear, "clearcoatNormal") ||
                    !check(gpu_material.Anisotropy.Texture, TextureColorSpace::Linear, "anisotropy") ||
                    !check(gpu_material.Iridescence.Texture, TextureColorSpace::Linear, "iridescence") ||
                    !check(gpu_material.Iridescence.ThicknessTexture, TextureColorSpace::Linear, "iridescenceThickness")
                ) {
                    return std::unexpected{std::move(resolve_result.error())};
                }
                return {};
            }();
            !result) {
            return std::unexpected{std::move(result.error())};
        }
        material_indices_by_gltf_material[material_index] = ctx.Buffers.Materials.Append(gpu_material);
        material_names.emplace_back(material_name);
    }
    const auto fallback_material_index = material_indices_by_gltf_material.empty() ? default_material_index : material_indices_by_gltf_material.back();
    if (!material_names.empty()) {
        R.patch<MaterialStore>(
            SceneEntity,
            [&](auto &store) {
                store.Names.insert(store.Names.end(), std::make_move_iterator(material_names.begin()), std::make_move_iterator(material_names.end()));
            }
        );
    }

    if (!new_pending_textures.empty()) {
        auto &pending = R.get_or_emplace<PendingTextureUploads>(SceneEntity);
        pending.Items.insert(pending.Items.end(), std::make_move_iterator(new_pending_textures.begin()), std::make_move_iterator(new_pending_textures.end()));
    }

    // Pre-reserve MeshStore arenas to avoid O(N) reallocations during bulk mesh creation.
    {
        auto plan = [&](const std::optional<::MeshData> &data) { if (data) ctx.Meshes.PlanCreate(*data); };
        for (const auto &scene_mesh : source_meshes) {
            if (scene_mesh.Triangles) {
                uint32_t morph_targets = scene_mesh.MorphData ? scene_mesh.MorphData->TargetCount : 0;
                ctx.Meshes.PlanCreate(*scene_mesh.Triangles, scene_mesh.TrianglePrimitives, scene_mesh.DeformData.has_value(), morph_targets);
            }
            plan(scene_mesh.Lines);
            plan(scene_mesh.Points);
        }
        ctx.Meshes.CommitReserves();
    }

    std::vector<entt::entity> mesh_entities;
    mesh_entities.reserve(source_meshes.size());
    // Per-mesh morph summary for later instance-component setup. Holds only what's needed
    // (TargetCount + DefaultWeights) — avoids deep-copying the full delta arrays.
    struct MorphSummary {
        uint32_t TargetCount{};
        std::vector<float> DefaultWeights;
    };
    std::vector<MorphSummary> mesh_morphs;
    mesh_morphs.reserve(source_meshes.size());
    entt::entity first_mesh_entity = entt::null;
    // Per-mesh: optional line entity + optional point entity
    struct ExtraPrimitiveEntities {
        entt::entity Lines{entt::null}, Points{entt::null};
    };
    std::vector<ExtraPrimitiveEntities> extra_entities_per_mesh(source_meshes.size());
    for (uint32_t mi = 0; mi < source_meshes.size(); ++mi) {
        auto &scene_mesh = source_meshes[mi];
        entt::entity mesh_entity = entt::null;
        if (scene_mesh.Triangles) {
            // Detect PBR extension features before material index remapping. `source_materials`
            // entries hold pre-remap gltf texture indices; the test below only checks for
            // InvalidSlot, which is identical pre/post remap.
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
            for (auto &local_material_index : scene_mesh.TrianglePrimitives.MaterialIndices) {
                local_material_index = local_material_index < material_indices_by_gltf_material.size() ?
                    material_indices_by_gltf_material[local_material_index] :
                    fallback_material_index;
            }
            // Snapshot per-primitive metadata + morph tangent deltas before move-into-CreateMesh
            // discards them. Stored as a sidecar so `BuildGltfScene` can re-emit verbatim. Layout-only
            // fields (VertexCounts/AttributeFlags/HasSourceIndices/VariantMappings) and TangentDeltas
            // aren't consumed by CreateMesh, so move them out of the source instead of copying.
            MorphSummary morph_summary;
            MeshSourceLayout layout{
                .VertexCounts = std::move(scene_mesh.TrianglePrimitives.VertexCounts),
                .AttributeFlags = std::move(scene_mesh.TrianglePrimitives.AttributeFlags),
                .HasSourceIndices = std::move(scene_mesh.TrianglePrimitives.HasSourceIndices),
                .VariantMappings = std::move(scene_mesh.TrianglePrimitives.VariantMappings),
                .Colors0ComponentCount = scene_mesh.TriangleAttrs.Colors0ComponentCount,
                .MorphTangentDeltas = scene_mesh.MorphData ? std::move(scene_mesh.MorphData->TangentDeltas) : std::vector<vec3>{},
            };
            if (scene_mesh.MorphData) {
                morph_summary = {scene_mesh.MorphData->TargetCount, scene_mesh.MorphData->DefaultWeights};
            }
            auto mesh = ctx.Meshes.CreateMesh(
                std::move(*scene_mesh.Triangles), std::move(scene_mesh.TriangleAttrs), std::move(scene_mesh.TrianglePrimitives),
                std::move(scene_mesh.DeformData), std::move(scene_mesh.MorphData)
            );
            const auto [me, _] = ::AddMesh(R, ctx.Meshes, SceneEntity, std::move(mesh), std::nullopt);
            mesh_entity = me;
            R.emplace<Path>(mesh_entity, source_path);
            R.emplace<SourceMeshIndex>(mesh_entity, mi);
            R.emplace<SourceMeshKind>(mesh_entity, MeshKind::Triangles);
            R.emplace<MeshSourceLayout>(mesh_entity, std::move(layout));
            if (!scene_mesh.Name.empty()) R.emplace<MeshName>(mesh_entity, scene_mesh.Name);
            if (mesh_pbr_mask != 0) R.emplace<PbrMeshFeatures>(mesh_entity, mesh_pbr_mask);
            mesh_morphs.emplace_back(std::move(morph_summary));
        } else {
            mesh_morphs.emplace_back();
        }
        if (first_mesh_entity == entt::null && mesh_entity != entt::null) first_mesh_entity = mesh_entity;
        mesh_entities.emplace_back(mesh_entity);

        auto create_extra = [&](std::optional<::MeshData> &data, ::MeshVertexAttributes &attrs, MeshKind kind) -> entt::entity {
            if (!data) return entt::null;
            auto m = ctx.Meshes.CreateMesh(std::move(*data), std::move(attrs), {});
            const auto [e, _] = ::AddMesh(R, ctx.Meshes, SceneEntity, std::move(m), std::nullopt);
            R.emplace<Path>(e, source_path);
            R.emplace<SourceMeshIndex>(e, mi);
            R.emplace<SourceMeshKind>(e, kind);
            if (!scene_mesh.Name.empty()) R.emplace<MeshName>(e, scene_mesh.Name);
            if (first_mesh_entity == entt::null) first_mesh_entity = e;
            return e;
        };
        auto lines_entity = create_extra(scene_mesh.Lines, scene_mesh.LineAttrs, MeshKind::Lines);
        auto points_entity = create_extra(scene_mesh.Points, scene_mesh.PointAttrs, MeshKind::Points);
        extra_entities_per_mesh[mi] = {lines_entity, points_entity};
    }

    const auto name_prefix = source_path.stem().string();
    R.patch<NameRegistry>(SceneEntity, [&](auto &registry) {
        registry.Names.reserve(registry.Names.size() + source_objects.size());
    });
    std::unordered_map<uint32_t, entt::entity> object_entities_by_node;
    object_entities_by_node.reserve(source_objects.size());
    std::unordered_map<uint32_t, std::vector<entt::entity>> skinned_mesh_instances_by_skin;
    skinned_mesh_instances_by_skin.reserve(asset.skins.size());
    std::unordered_map<uint32_t, entt::entity> armature_data_entities_by_skin;
    armature_data_entities_by_skin.reserve(asset.skins.size());

    std::vector<entt::entity> all_imported_objects;
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
            if (mi < extra_entities_per_mesh.size()) {
                const auto &[lines, points] = extra_entities_per_mesh[mi];
                return lines != entt::null ? lines : points;
            }
            return entt::null;
        }();
        if (primary_mesh_entity != entt::null) {
            object_entity = ::AddMeshInstance(
                R, SceneEntity,
                primary_mesh_entity,
                {.Name = object_name, .Transform = object.LocalTransform, .Select = MeshInstanceCreateInfo::SelectBehavior::None, .Visible = true}
            );
        } else if (object.ObjectType == gltf::Object::Type::Camera && object.CameraIndex && *object.CameraIndex < asset.cameras.size()) {
            const auto &cam = asset.cameras[*object.CameraIndex];
            object_entity = ::AddCamera(R, ctx.Meshes, ctx.Buffers, SceneEntity, {.Name = object_name, .Transform = object.LocalTransform, .Select = MeshInstanceCreateInfo::SelectBehavior::None});
            R.replace<::Camera>(object_entity, ConvertCamera(cam));
            R.emplace<SourceCameraIndex>(object_entity, *object.CameraIndex);
            if (!cam.name.empty()) R.emplace<CameraName>(object_entity, std::string{cam.name});
        } else if (object.ObjectType == gltf::Object::Type::Light && object.LightIndex && *object.LightIndex < asset.lights.size()) {
            const auto &light = asset.lights[*object.LightIndex];
            object_entity = ::AddLight(R, ctx.Meshes, ctx.Buffers, SceneEntity, {.Name = object_name, .Transform = object.LocalTransform, .Select = MeshInstanceCreateInfo::SelectBehavior::None}, ConvertLight(light));
            R.emplace<SourceLightIndex>(object_entity, *object.LightIndex);
            if (!light.name.empty()) R.emplace<LightName>(object_entity, std::string{light.name});
        } else {
            object_entity = ::AddEmpty(R, ctx.Meshes, ctx.Buffers, SceneEntity, {.Name = object_name, .Transform = object.LocalTransform, .Select = MeshInstanceCreateInfo::SelectBehavior::None});
        }
        // Companion instances for non-triangle primitives, parented under primary with identity local.
        if (object.ObjectType == gltf::Object::Type::Mesh && object.MeshIndex && *object.MeshIndex < extra_entities_per_mesh.size()) {
            const auto &extras = extra_entities_per_mesh[*object.MeshIndex];
            for (const auto extra_entity : {extras.Lines, extras.Points}) {
                if (extra_entity == entt::null || extra_entity == primary_mesh_entity) continue;
                const auto extra_instance = ::AddMeshInstance(
                    R, SceneEntity,
                    extra_entity,
                    {.Name = object_name, .Transform = Transform{}, .Select = MeshInstanceCreateInfo::SelectBehavior::None, .Visible = true}
                );
                SetParent(R, extra_instance, object_entity);
            }
        }

        object_entities_by_node[object.NodeIndex] = object_entity;
        R.emplace<SourceNodeIndex>(object_entity, object.NodeIndex);
        // `object.Name` was already synthesized by the loader's MakeNodeName when source was
        // empty; `asset.nodes[i].name` preserves the raw value. Capture source-empty / collision-renamed.
        if (object.NodeIndex < asset.nodes.size()) {
            const std::string raw_name(asset.nodes[object.NodeIndex].name);
            if (raw_name.empty()) R.emplace<SourceEmptyName>(object_entity);
            else if (const auto *n = R.try_get<const Name>(object_entity); n && n->Value != raw_name) {
                R.emplace<SourceObjectName>(object_entity, SourceObjectName{raw_name});
            }
        }
        all_imported_objects.emplace_back(object_entity);
        // glTF node.skin is deform linkage, not a transform-parent relationship.
        if (object.SkinIndex && R.all_of<Instance>(object_entity)) skinned_mesh_instances_by_skin[*object.SkinIndex].emplace_back(object_entity);
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
        if (parent_it == object_entities_by_node.end()) continue;
        SetParent(R, child_it->second, parent_it->second);
    }

    // Stubs for out-of-scene nodes (referenced only by non-default scenes) so build emits them
    // like the file round-trip does. They carry only what build needs and aren't in
    // `object_entities_by_node`, so runtime systems that walk the scene tree don't see them.
    for (uint32_t node_index = 0; node_index < asset.nodes.size(); ++node_index) {
        if (traversal.InScene[node_index]) continue;
        const auto &source_node = asset.nodes[node_index];
        const auto e = R.create();
        R.emplace<SourceNodeIndex>(e, node_index);
        R.emplace<Transform>(e, local_transforms[node_index]);
        R.emplace<WorldTransform>(e);
        if (const auto mesh_index = ToIndex(source_node.meshIndex, asset.meshes.size());
            mesh_index && *mesh_index < mesh_entities.size() && mesh_entities[*mesh_index] != entt::null) {
            R.emplace<Instance>(e, mesh_entities[*mesh_index]);
        }
        if (source_node.name.empty()) R.emplace<SourceEmptyName>(e);
        else R.emplace<Name>(e, std::string{source_node.name});
    }

    // KHR_physics_rigid_bodies: collision filter entities are created here (the system-name dedupe
    // map needs to live across all filters). Materials/joint-defs were already promoted above.
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
                    it->second = R.create();
                    R.emplace<CollisionSystem>(it->second, CollisionSystem{.Name = it->first});
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
            const auto e = R.create();
            R.emplace<CollisionFilter>(e, CollisionFilter{.Systems = resolve_systems(src.collisionSystems), .Mode = mode, .CollideSystems = std::move(collide_systems)});
            R.emplace<SourceCollisionFilterIndex>(e, i);
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
                    if (R.all_of<Instance>(entity)) return R.get<const Instance>(entity).Entity;
                    return null_entity;
                }();
                R.emplace<ColliderShape>(entity, ColliderShape{.Shape = node.Collider->Shape, .MeshEntity = collider_mesh_entity});
                // Imported collider state is authoritative — engine must not auto-derive over it.
                R.emplace<ColliderPolicy>(entity, ColliderPolicy{.AutoFitDims = false, .LockedKind = true});
                if (node.Material) {
                    R.replace<ColliderMaterial>(
                        entity,
                        ColliderMaterial{
                            .PhysicsMaterialEntity = resolve_mat(node.Material->PhysicsMaterialIndex),
                            .CollisionFilterEntity = resolve_filter(node.Material->CollisionFilterIndex),
                        }
                    );
                }
            }
            if (node.Motion) {
                R.emplace<PhysicsMotion>(entity, *node.Motion);
                if (node.Velocity) R.replace<PhysicsVelocity>(entity, *node.Velocity);
            }
            if (node.Trigger) {
                const auto &td = *node.Trigger;
                if (td.Shape) {
                    // GeometryTrigger: reuse ColliderShape + TriggerTag. Skip if a solid collider
                    // already took this entity — KHR declares nodes as one-or-the-other.
                    if (!R.all_of<ColliderShape>(entity)) {
                        const auto trigger_mesh_entity = (td.GeometryMeshIndex && *td.GeometryMeshIndex < mesh_entities.size()) ? mesh_entities[*td.GeometryMeshIndex] : null_entity;
                        R.emplace<ColliderShape>(entity, ColliderShape{.Shape = *td.Shape, .MeshEntity = trigger_mesh_entity});
                        R.emplace<ColliderPolicy>(entity, ColliderPolicy{.AutoFitDims = false, .LockedKind = true});
                        R.emplace<TriggerTag>(entity);
                        R.patch<ColliderMaterial>(entity, [&](auto &m) { m.CollisionFilterEntity = resolve_filter(td.CollisionFilterIndex); });
                    }
                } else {
                    // NodesTrigger: compound zone.
                    std::vector<entt::entity> resolved_nodes;
                    resolved_nodes.reserve(td.NodeIndices.size());
                    for (const auto node_idx : td.NodeIndices) {
                        auto nit = object_entities_by_node.find(node_idx);
                        resolved_nodes.emplace_back(nit != object_entities_by_node.end() ? nit->second : entt::null);
                    }
                    R.emplace<TriggerNodes>(entity, TriggerNodes{.Nodes = std::move(resolved_nodes), .CollisionFilterEntity = resolve_filter(td.CollisionFilterIndex)});
                }
            }
            if (node.Joint) {
                const auto &jd = *node.Joint;
                auto nit = object_entities_by_node.find(jd.ConnectedNodeIndex);
                const auto def_entity = jd.JointDefIndex < physics_jointdef_entities.size() ? physics_jointdef_entities[jd.JointDefIndex] : null_entity;
                R.emplace<PhysicsJoint>(
                    entity,
                    PhysicsJoint{.ConnectedNode = nit != object_entities_by_node.end() ? nit->second : entt::null, .JointDefEntity = def_entity, .EnableCollision = jd.EnableCollision}
                );
            }
        }
    }

    // Build + consume each skin in a single pass. Skin construction needs `parents`,
    // `local_transforms`, `traversal`, `nearest_object_ancestor` (all set up earlier) and the
    // already-emitted `object_entities_by_node` / `scene_nodes_by_index` (set up just above).
    std::unordered_set<uint32_t> joint_node_indices;
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
        // Deterministic armature source anchor: explicit skin.skeleton if present,
        // otherwise the computed joint ancestry root. Do not synthesize extra roots.
        const auto anchor_node_index = skeleton_node_index ? skeleton_node_index : ComputeCommonAncestor(*ordered_joint_nodes, parents);
        const auto parent_object_node_index = (anchor_node_index && traversal.InScene[*anchor_node_index]) ? nearest_object_ancestor[*anchor_node_index] : std::optional<uint32_t>{};
        auto inverse_bind_matrices = LoadInverseBindMatrices(asset, skin, uint32_t(ordered_joint_nodes->size()));

        const auto armature_data_entity = R.create();
        auto &armature = R.emplace<Armature>(armature_data_entity);
        armature_data_entities_by_skin[skin_index] = armature_data_entity;

        ArmatureImportedSkin imported_skin{
            .SkinIndex = skin_index,
            .SkeletonNodeIndex = skeleton_node_index,
            .AnchorNodeIndex = anchor_node_index,
            .OrderedJointNodeIndices = {},
            .InverseBindMatrices = std::move(inverse_bind_matrices),
        };
        imported_skin.OrderedJointNodeIndices.reserve(ordered_joint_nodes->size());

        std::unordered_map<uint32_t, BoneId> joint_node_to_bone_id;
        joint_node_to_bone_id.reserve(ordered_joint_nodes->size());
        for (const auto joint_node_index : *ordered_joint_nodes) {
            joint_node_indices.emplace(joint_node_index);
            const auto parent_joint_node_index = joint_parent_map.at(joint_node_index);
            auto rest_local = ComputeJointRestLocal(skin_index, joint_node_index, parent_joint_node_index, anchor_node_index, parents, local_transforms);
            if (!rest_local) return std::unexpected{rest_local.error()};

            std::optional<BoneId> parent_bone_id;
            if (parent_joint_node_index) {
                if (const auto parent_it = joint_node_to_bone_id.find(*parent_joint_node_index);
                    parent_it != joint_node_to_bone_id.end()) {
                    parent_bone_id = parent_it->second;
                }
            }

            const auto source_name = MakeNodeName(asset, joint_node_index);
            const auto joint_name = source_name.empty() ? std::format("Joint{}", joint_node_index) : source_name;
            const auto bone_id = armature.AddBone(joint_name, parent_bone_id, *rest_local, joint_node_index);
            joint_node_to_bone_id.emplace(joint_node_index, bone_id);
            if (const auto object_it = object_entities_by_node.find(joint_node_index);
                object_it != object_entities_by_node.end() &&
                R.all_of<Instance>(object_it->second) &&
                !R.all_of<PhysicsMotion>(object_it->second) &&
                !R.all_of<BoneAttachment>(object_it->second)) {
                R.emplace<BoneAttachment>(object_it->second, armature_data_entity, bone_id);
            }
            imported_skin.OrderedJointNodeIndices.emplace_back(joint_node_index);
        }
        imported_skin.InverseBindMatrices.resize(imported_skin.OrderedJointNodeIndices.size(), I4);
        armature.ImportedSkin = std::move(imported_skin);
        armature.FinalizeStructure();
        if (!anchor_node_index) {
            return std::unexpected{std::format("glTF import failed for '{}': skin {} has no deterministic anchor node.", source_path.string(), skin_index)};
        }

        if (*anchor_node_index >= asset.nodes.size() || !traversal.InScene[*anchor_node_index]) {
            return std::unexpected{std::format("glTF import failed for '{}': skin {} anchor node {} is not in the imported scene.", source_path.string(), skin_index, *anchor_node_index)};
        }

        const std::string skin_name(skin.name);
        const auto armature_entity = R.create();
        R.emplace<ObjectKind>(armature_entity, ObjectType::Armature);
        R.emplace<ArmatureObject>(armature_entity, armature_data_entity);
        const auto t = ToTransform(traversal.WorldTransforms[*anchor_node_index]);
        R.emplace<Transform>(armature_entity, t);
        R.emplace<WorldTransform>(armature_entity, t);
        R.emplace<Name>(armature_entity, ::CreateName(R, SceneEntity, skin_name.empty() ? std::format("{}_Armature{}", name_prefix, skin_index) : skin_name));
        if (skin_name.empty()) R.emplace<SourceEmptyName>(armature_entity);
        else R.emplace<SkinName>(armature_entity, skin_name);

        if (parent_object_node_index) {
            if (const auto parent_it = object_entities_by_node.find(*parent_object_node_index);
                parent_it != object_entities_by_node.end()) {
                SetParentKeepWorld(R, armature_entity, parent_it->second);
            }
        }

        all_imported_objects.emplace_back(armature_entity);
        if (first_armature_entity == entt::null) first_armature_entity = armature_entity;
        if (first_object_entity == entt::null) first_object_entity = armature_entity;

        if (const auto skinned_it = skinned_mesh_instances_by_skin.find(skin_index);
            skinned_it != skinned_mesh_instances_by_skin.end()) {
            for (const auto mesh_instance_entity : skinned_it->second) {
                if (!R.valid(mesh_instance_entity) || !R.all_of<Instance>(mesh_instance_entity)) continue;
                R.emplace_or_replace<ArmatureModifier>(mesh_instance_entity, armature_data_entity, armature_entity);
                SetParentKeepWorld(R, mesh_instance_entity, armature_entity);
            }
        } else {
            return std::unexpected{std::format("glTF import failed '{}': skin {} is used but no mesh instances were emitted for skin binding.", source_path.string(), skin_index)};
        }

        // Create pose state — GPU deform buffer allocation deferred to ProcessComponentEvents.
        const auto bone_count = armature.Bones.size();
        R.emplace<ArmaturePoseState>(
            armature_data_entity,
            ArmaturePoseState{
                .BonePoseDelta = std::vector<Transform>(bone_count),
                .BoneUserOffset = std::vector<Transform>(bone_count),
                .BonePoseWorld = std::vector<mat4>(bone_count, I4),
                .GpuDeformRange = {},
            }
        );
        ::CreateBoneInstances(R, ctx.Meshes, SceneEntity, armature_entity, armature_data_entity);
        // Mark each bone entity with its source joint NodeIndex (for SaveScene round-trip).
        const auto &bone_entities_for_source = R.get<const ArmatureObject>(armature_entity).BoneEntities;
        for (uint32_t i = 0; i < armature.Bones.size(); ++i) {
            const auto joint_node_index = armature.Bones[i].JointNodeIndex;
            if (!joint_node_index) continue;
            R.emplace<SourceNodeIndex>(bone_entities_for_source[i], *joint_node_index);
            if (*joint_node_index < asset.nodes.size() && asset.nodes[*joint_node_index].name.empty()) {
                R.emplace<SourceEmptyName>(bone_entities_for_source[i]);
            }
        }

        // Auto-wire Child Of on bones whose joint node has a physics-driven ancestor object,
        // so the skin follows simulated motion when an asset pairs rigid bodies with skinning via the scene graph.
        // Target is the nearest ancestor object with PhysicsMotion; InverseMatrix bakes the rest offset.
        {
            const auto find_physics_ancestor_entity = [&](uint32_t node_index) -> entt::entity {
                for (std::optional<uint32_t> cur = node_index; cur;) {
                    if (const auto oit = object_entities_by_node.find(*cur);
                        oit != object_entities_by_node.end() && R.all_of<PhysicsMotion>(oit->second)) return oit->second;
                    if (*cur >= parents.size()) break;
                    cur = parents[*cur];
                }
                return entt::null;
            };
            const auto &arm_obj = R.get<const ArmatureObject>(armature_entity);
            const mat4 armature_world = ToMatrix(R.get<const WorldTransform>(armature_entity));
            for (uint32_t i = 0; i < armature.Bones.size(); ++i) {
                const auto &bone = armature.Bones[i];
                if (!bone.JointNodeIndex) continue;
                const auto target = find_physics_ancestor_entity(*bone.JointNodeIndex);
                if (target == entt::null) continue;
                R.emplace<BoneConstraints>(
                    arm_obj.BoneEntities[i],
                    BoneConstraints{
                        .Stack = {BoneConstraint{
                            .TargetEntity = target,
                            .Influence = 1.f,
                            .Data = ChildOfData{.InverseMatrix = glm::inverse(ToMatrix(R.get<const WorldTransform>(target))) * (armature_world * bone.RestWorld)},
                        }}
                    }
                );
            }
        }
    }

    // Per source-derived entity: tag with source parent / sibling position / matrix-form flag.
    for (const auto [entity, sni] : R.view<const SourceNodeIndex>().each()) {
        if (sni.Value >= asset.nodes.size()) continue;
        if (const auto parent_idx = parents[sni.Value]) {
            R.emplace<SourceParentNodeIndex>(entity, *parent_idx);
            // Sibling position in parent's bounds-filtered children list.
            uint32_t sibling_idx = 0;
            for (const auto child_raw : asset.nodes[*parent_idx].children) {
                const auto child = ToIndex(child_raw, asset.nodes.size());
                if (!child) continue;
                if (*child == sni.Value) {
                    R.emplace<SourceSiblingIndex>(entity, sibling_idx);
                    break;
                }
                ++sibling_idx;
            }
        }
        if (source_matrices[sni.Value]) R.emplace<SourceMatrixTransform>(entity, *source_matrices[sni.Value]);
    }

    std::unordered_map<uint32_t, std::vector<std::pair<entt::entity, BoneId>>> armature_targets_by_joint_node;
    for (const auto &entry : armature_data_entities_by_skin) {
        const auto armature_data_entity = entry.second;
        const auto &armature = R.get<const Armature>(armature_data_entity);
        for (const auto &bone : armature.Bones) {
            if (!bone.JointNodeIndex) continue;
            armature_targets_by_joint_node[*bone.JointNodeIndex].emplace_back(armature_data_entity, bone.Id);
        }
    }

    // Set up morph weight state for mesh instances with morph targets.
    // GPU allocation deferred to ProcessComponentEvents (GpuWeightRange left as default {0,0}).
    // Build a map: node_index -> mesh instance entity, for resolving weight animation channels.
    std::unordered_map<uint32_t, entt::entity> morph_instance_by_node;
    for (const auto &object : source_objects) {
        if (object.ObjectType != gltf::Object::Type::Mesh || !object.MeshIndex) continue;
        if (*object.MeshIndex >= mesh_morphs.size()) continue;
        const auto obj_it = object_entities_by_node.find(object.NodeIndex);
        if (obj_it == object_entities_by_node.end()) continue;
        const auto instance_entity = obj_it->second;
        if (!R.all_of<Instance>(instance_entity)) continue;

        const auto &morph = mesh_morphs[*object.MeshIndex];
        if (morph.TargetCount == 0) continue;

        auto weights = [&] {
            if (!object.NodeWeights) return morph.DefaultWeights;
            std::vector<float> w(morph.TargetCount, 0.f);
            std::copy_n(object.NodeWeights->begin(), std::min(uint32_t(object.NodeWeights->size()), morph.TargetCount), w.begin());
            return w;
        }();
        R.emplace<MorphWeightState>(instance_entity, MorphWeightState{.Weights = std::move(weights), .GpuWeightRange = {}});
        morph_instance_by_node[object.NodeIndex] = instance_entity;
    }

    // Resolve object/node transform animations (empties, meshes, cameras, lights).
    // Channels targeting skin joints are handled by ArmatureAnimation and skipped here.
    std::unordered_map<entt::entity, Transform> node_anim_bindings;
    node_anim_bindings.reserve(object_entities_by_node.size());
    for (const auto &[node_index, object_entity] : object_entities_by_node) {
        if (!R.valid(object_entity) || node_index >= local_transforms.size()) continue;
        node_anim_bindings.emplace(object_entity, local_transforms[node_index]);
    }

    bool imported_animation = false;
    const auto append_armature_clip = [&](entt::entity target_data_entity, ::AnimationClip &&resolved_clip) {
        if (resolved_clip.Channels.empty()) return;
        imported_animation = true;
        if (auto *existing = R.try_get<ArmatureAnimation>(target_data_entity)) {
            existing->Clips.emplace_back(std::move(resolved_clip));
        } else {
            R.emplace<ArmatureAnimation>(target_data_entity, ArmatureAnimation{.Clips = {std::move(resolved_clip)}});
        }
    };
    const auto append_morph_clip = [&](entt::entity instance_entity, MorphWeightClip &&resolved_clip) {
        if (resolved_clip.Channels.empty()) return;
        imported_animation = true;
        if (auto *existing = R.try_get<MorphWeightAnimation>(instance_entity)) {
            existing->Clips.emplace_back(std::move(resolved_clip));
        } else {
            R.emplace<MorphWeightAnimation>(instance_entity, MorphWeightAnimation{.Clips = {std::move(resolved_clip)}});
        }
    };
    const auto append_node_clip = [&](entt::entity object_entity, ::AnimationClip &&resolved_clip) {
        if (resolved_clip.Channels.empty()) return;
        imported_animation = true;
        if (auto *existing = R.try_get<NodeTransformAnimation>(object_entity)) {
            existing->Clips.emplace_back(std::move(resolved_clip));
            return;
        }
        const auto binding_it = node_anim_bindings.find(object_entity);
        if (binding_it == node_anim_bindings.end()) return;
        R.emplace<NodeTransformAnimation>(
            object_entity,
            NodeTransformAnimation{.Clips = {std::move(resolved_clip)}, .ActiveClipIndex = 0, .RestLocal = binding_it->second}
        );
    };

    // Parse + resolve animations in a single pass: each source channel goes straight from
    // fastgltf accessors into the per-entity ECS clip it targets. AnimationOrder records the
    // source-form names of animations that produced at least one valid channel.
    std::vector<std::string> animation_order;
    animation_order.reserve(asset.animations.size());
    struct ChannelTargetSpec {
        AnimationPath Path;
        std::size_t ComponentCount;
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
                values.resize(output_accessor.count);
                fastgltf::copyFromAccessor<float>(asset, output_accessor, values.data());
            } else {
                values.resize(output_accessor.count * target_spec->ComponentCount);
                if (target_spec->ComponentCount == 4) fastgltf::copyFromAccessor<vec4>(asset, output_accessor, reinterpret_cast<vec4 *>(values.data()));
                else fastgltf::copyFromAccessor<vec3>(asset, output_accessor, reinterpret_cast<vec3 *>(values.data()));
            }

            if (!times.empty()) max_time = std::max(max_time, times.back());
            any_channel = true;
            const uint32_t target_node_index = uint32_t(*channel.nodeIndex);

            if (target_spec->Path == AnimationPath::Weights) {
                const auto inst_it = morph_instance_by_node.find(target_node_index);
                if (inst_it == morph_instance_by_node.end()) continue;
                auto &resolved_clip = morph_clips_by_entity
                                          .try_emplace(inst_it->second, MorphWeightClip{.Name = anim_name, .DurationSeconds = 0.f, .Channels = {}})
                                          .first->second;
                resolved_clip.Channels.emplace_back(MorphWeightChannel{
                    .Interp = interp,
                    .TimesSeconds = std::move(times),
                    .Values = std::move(values),
                });
                continue;
            }

            if (const auto armature_it = armature_targets_by_joint_node.find(target_node_index);
                armature_it != armature_targets_by_joint_node.end()) {
                for (const auto &[target_data_entity, bone_id] : armature_it->second) {
                    const auto &armature = R.get<const Armature>(target_data_entity);
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
            if (object_it == object_entities_by_node.end() || !R.valid(object_it->second)) continue;

            auto &resolved_clip = node_clips_by_entity
                                      .try_emplace(object_it->second, ::AnimationClip{.Name = anim_name, .DurationSeconds = 0.f, .Channels = {}})
                                      .first->second;
            resolved_clip.Channels.emplace_back(::AnimationChannel{.BoneIndex = 0, .Target = target_spec->Path, .Interp = interp, .TimesSeconds = std::move(times), .Values = std::move(values)});
        }

        if (!any_channel) continue;
        animation_order.emplace_back(std::move(anim_name));

        for (auto &[_, c] : armature_clips_by_entity) c.DurationSeconds = max_time;
        for (auto &[_, c] : morph_clips_by_entity) c.DurationSeconds = max_time;
        for (auto &[_, c] : node_clips_by_entity) c.DurationSeconds = max_time;

        for (auto &[target_data_entity, resolved_clip] : armature_clips_by_entity) append_armature_clip(target_data_entity, std::move(resolved_clip));
        for (auto &[instance_entity, resolved_clip] : morph_clips_by_entity) append_morph_clip(instance_entity, std::move(resolved_clip));
        for (auto &[object_entity, resolved_clip] : node_clips_by_entity) append_node_clip(object_entity, std::move(resolved_clip));
    }
    R.patch<gltf::SourceAssets>(SceneEntity, [&](auto &a) { a.AnimationOrder = std::move(animation_order); });

    { // Get timeline range from imported animation durations
        float max_dur = 0;
        for (const auto [_, anim] : R.view<const ArmatureAnimation>().each()) {
            for (const auto &clip : anim.Clips) max_dur = std::max(max_dur, clip.DurationSeconds);
        }
        for (const auto [_, anim] : R.view<const MorphWeightAnimation>().each()) {
            for (const auto &clip : anim.Clips) max_dur = std::max(max_dur, clip.DurationSeconds);
        }
        for (const auto [_, anim] : R.view<const NodeTransformAnimation>().each()) {
            for (const auto &clip : anim.Clips) max_dur = std::max(max_dur, clip.DurationSeconds);
        }
        if (max_dur > 0) R.patch<AnimationTimeline>(SceneEntity, [&](auto &tl) { tl.EndFrame = int(std::ceil(max_dur * tl.Fps)); });
    }

    if (source_ibl) {
        if (auto *prev = R.try_get<PendingEnvironmentImport>(SceneEntity)) prev_pending_env_backup = *prev;
        const auto [diffuse_slot, specular_slot] = AllocateIblCubeSlots(ctx.Slots);
        R.emplace_or_replace<PendingEnvironmentImport>(SceneEntity, *source_ibl, diffuse_slot, specular_slot);
        replaced_pending_env = true;
    }

    const auto active_entity =
        first_camera_object_entity != entt::null ? first_camera_object_entity :
        first_mesh_object_entity != entt::null   ? first_mesh_object_entity :
        first_armature_entity != entt::null      ? first_armature_entity :
        first_root_empty_entity != entt::null    ? first_root_empty_entity :
                                                   first_object_entity;
    R.clear<Active, Selected>();
    if (active_entity != entt::null) R.emplace<Active>(active_entity);
    for (const auto e : all_imported_objects) R.emplace<Selected>(e);
    import_rollback_guard.Enabled = false;

    return gltf::PopulateResult{.Active = active_entity, .FirstMesh = first_mesh_entity, .FirstCameraObject = first_camera_object_entity, .ImportedAnimation = imported_animation};
}

std::expected<void, std::string> SaveGltf(const SaveContext &sc, const std::filesystem::path &path) {
    const Timer timer{"SaveGltf"};

    const auto &r = sc.R;
    const auto &meshes = sc.Meshes;

    // Order entities in `view` by their `TIndex` sidecar value. Entities without `TIndex`
    // (runtime-added) land after the source range. Used for cameras, lights, physics resources.
    const auto ordered_by_source = [&]<typename TIndex>(auto view) {
        std::vector<std::pair<uint32_t, entt::entity>> ordered;
        uint32_t next = 0;
        for (const auto e : view) {
            if (const auto *si = r.try_get<const TIndex>(e)) {
                ordered.emplace_back(si->Value, e);
                next = std::max(next, si->Value + 1u);
            }
        }
        for (const auto e : view) {
            if (!r.all_of<TIndex>(e)) ordered.emplace_back(next++, e);
        }
        std::ranges::sort(ordered, {}, &std::pair<uint32_t, entt::entity>::first);
        return ordered;
    };

    // Source-form scene metadata + texture/image/sampler arrays come from the gltf::SourceAssets
    // sidecar — encoded image bytes, sampler-config collapse, and asset.* metadata aren't
    // recoverable from registry/GPU state. Cameras and lights emit below from per-entity
    // components (see CameraName/LightName). Materials emit directly from PBRMaterial (GPU buffer)
    // + per-material `MaterialSourceMeta` delta, no intermediate type.
    const auto *src_assets = r.try_get<const gltf::SourceAssets>(sc.SceneEntity);
    // Source-form scene-level metadata (Copyright/Generator/MinVersion/Asset.*, DefaultScene,
    // ExtensionsRequired/MaterialVariants/ExtrasByEntity, Textures/Images/Samplers/IBL) is read
    // directly from `src_assets` at each emit site below (no intermediate aggregate).
    static const gltf::SourceAssets EmptySourceAssets{};
    const auto &sa = src_assets ? *src_assets : EmptySourceAssets;
    // Materials: skip the engine "Default" at registry index 0; loaded gltf materials live at
    // [1, count). `MaterialSourceMeta` carries the round-trip-only deltas (ext-block presence,
    // source texture indices, KHR_texture_transform meta, KHR_materials_emissive_strength split).
    const auto &names = r.get<const MaterialStore>(sc.SceneEntity).Names;
    const auto material_count = sc.Buffers.Materials.Count();
    const auto &material_metas = src_assets ? src_assets->MaterialMetas : std::vector<MaterialSourceMeta>{};

    // Mesh entities → MeshIndex. Source meshes use SourceMeshIndex for stable round-trip ordering;
    // engine-generated meshes (no SourceMeshIndex) are skipped — they don't belong in save_meshes.
    std::unordered_map<entt::entity, uint32_t> mesh_entity_to_index;
    uint32_t source_mesh_count = 0;
    auto source_mesh_view = r.view<const Mesh, const SourceMeshIndex>();
    for (const auto e : source_mesh_view) {
        const auto smi_value = source_mesh_view.get<const SourceMeshIndex>(e).Value;
        mesh_entity_to_index[e] = smi_value;
        source_mesh_count = std::max(source_mesh_count, smi_value + 1u);
    }

    // Group source-mesh entities by SourceMeshIndex (one Triangles/Lines/Points entity per slot).
    // The fastgltf::Mesh emit pass below reads vertex/face/skin/morph data straight from
    // MeshStore for each grouped entity — no per-mesh `gltf::MeshData` aggregate is needed.
    struct MeshEntitySet {
        entt::entity Triangles{entt::null}, Lines{entt::null}, Points{entt::null};
        std::string Name;
    };
    std::vector<MeshEntitySet> mesh_groups(source_mesh_count);
    for (const auto [entity, smi, kind] : r.view<const SourceMeshIndex, const SourceMeshKind>().each()) {
        if (!r.all_of<Mesh>(entity)) continue;
        auto &g = mesh_groups[smi.Value];
        if (kind.Value == MeshKind::Triangles) g.Triangles = entity;
        else if (kind.Value == MeshKind::Lines) g.Lines = entity;
        else g.Points = entity;
        if (g.Name.empty()) {
            if (const auto *mn = r.try_get<const MeshName>(entity)) g.Name = mn->Value;
        }
    }

    // Cameras / lights: one entry per component-bearing entity, in source-aligned order. Per the
    // Khronos sample set source cameras/lights aren't shared across nodes, so 1:1 matches source counts.
    // Store the entities here; emit to fastgltf::Asset later (when `asset` exists).
    std::unordered_map<entt::entity, uint32_t> camera_entity_to_index, light_entity_to_index;
    std::vector<entt::entity> camera_entities_ordered, light_entities_ordered;
    {
        auto camera_view = r.view<const ::Camera>();
        for (const auto &[_, entity] : ordered_by_source.operator()<SourceCameraIndex>(camera_view)) {
            camera_entity_to_index[entity] = uint32_t(camera_entities_ordered.size());
            camera_entities_ordered.emplace_back(entity);
        }
        auto light_view = r.view<const PunctualLight>();
        for (const auto &[_, entity] : ordered_by_source.operator()<SourceLightIndex>(light_view)) {
            light_entity_to_index[entity] = uint32_t(light_entities_ordered.size());
            light_entities_ordered.emplace_back(entity);
        }
    }

    // Armature data entities → SkinIndex (source-fidelity from ImportedSkin if present).
    std::unordered_map<entt::entity, uint32_t> armature_data_to_skin_index;
    for (const auto e : r.view<const Armature>()) {
        const auto &arm = r.get<const Armature>(e);
        armature_data_to_skin_index[e] = arm.ImportedSkin ? arm.ImportedSkin->SkinIndex : uint32_t(armature_data_to_skin_index.size());
    }

    // Map data entity → ArmatureObject entity (for parent-of-armature lookup + name).
    std::unordered_map<entt::entity, entt::entity> armature_data_to_object;
    for (const auto [e, ao] : r.view<const ArmatureObject>().each()) armature_data_to_object[ao.Entity] = e;

    // Scene tree: only source-derived entities (with `SourceNodeIndex`) become nodes; engine
    // helpers like the armature object are skipped. Hierarchy comes from `SourceParentNodeIndex`,
    // not `SceneNode` (which gets mutated by skinning/armature re-parenting at populate).
    std::unordered_map<entt::entity, uint32_t> entity_to_node_index;
    uint32_t total_node_count = 0;
    for (const auto [e, sni] : r.view<const SourceNodeIndex>().each()) {
        entity_to_node_index[e] = sni.Value;
        total_node_count = std::max(total_node_count, sni.Value + 1u);
    }
    // Children paired with sibling position so we sort in source order.
    std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, uint32_t>>> children_by_parent;
    for (const auto [e, sni, spi] : r.view<const SourceNodeIndex, const SourceParentNodeIndex>().each()) {
        const auto *ssi = r.try_get<const SourceSiblingIndex>(e);
        children_by_parent[spi.Value].emplace_back(ssi ? ssi->Value : sni.Value, sni.Value);
    }
    for (auto &[_, kids] : children_by_parent) std::ranges::sort(kids, {}, &std::pair<uint32_t, uint32_t>::first);

    // node_index → entity (or entt::null when a SourceNodeIndex value has gaps). The node
    // emission loop below reads each entity's components directly — no parallel `gltf::Node`
    // aggregate needed on save.
    std::vector<entt::entity> node_to_entity(total_node_count, entt::null);
    for (const auto [entity, node_index] : entity_to_node_index) {
        if (node_index < node_to_entity.size()) node_to_entity[node_index] = entity;
    }

    // Per node: world transforms of every Object that targets that node — used for
    // EXT_mesh_gpu_instancing fan-in (multi-object → single node).
    std::vector<std::vector<Transform>> node_instance_worlds(total_node_count);
    auto object_view = r.view<const Transform, const ObjectKind>();
    for (const auto entity : object_view) {
        if (object_view.get<const ObjectKind>(entity).Value == ObjectType::Armature) continue; // → gltf::Skin, handled separately.
        const auto it = entity_to_node_index.find(entity);
        if (it == entity_to_node_index.end() || it->second >= total_node_count) continue;
        node_instance_worlds[it->second].emplace_back(r.get<const WorldTransform>(entity));
    }

    // Asset is declared early so collision filters can be emitted directly into
    // `asset.collisionFilters` (their pmr allocator binds to the asset).
    fastgltf::Asset asset;

    // Physics document-level resources, source-aligned via the per-resource index sidecars.
    // Emit fastgltf entries directly into asset.* — the per-entity index map lets node fan-in
    // resolve refs without a parallel staging vector.
    std::unordered_map<entt::entity, uint32_t> physics_material_to_index, physics_jointdef_to_index, collision_filter_to_index;
    {
        auto mat_view = r.view<const PhysicsMaterial>();
        for (const auto &[_, e] : ordered_by_source.operator()<SourcePhysicsMaterialIndex>(mat_view)) {
            const auto &pm = mat_view.get<const PhysicsMaterial>(e);
            physics_material_to_index[e] = uint32_t(asset.physicsMaterials.size());
            asset.physicsMaterials.emplace_back(fastgltf::PhysicsMaterial{
                .staticFriction = pm.StaticFriction,
                .dynamicFriction = pm.DynamicFriction,
                .restitution = pm.Restitution,
                .frictionCombine = FromCombine(pm.FrictionCombine),
                .restitutionCombine = FromCombine(pm.RestitutionCombine),
            });
        }
        auto jd_view = r.view<const ::PhysicsJointDef>();
        for (const auto &[_, e] : ordered_by_source.operator()<SourcePhysicsJointDefIndex>(jd_view)) {
            const auto &jd = jd_view.get<const ::PhysicsJointDef>(e);
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
            physics_jointdef_to_index[e] = uint32_t(asset.physicsJoints.size());
            asset.physicsJoints.emplace_back(fastgltf::PhysicsJoint{.limits = std::move(limits), .drives = std::move(drives)});
        }
        const auto resolve_system_names = [&](std::span<const entt::entity> systems) {
            fastgltf::pmr::MaybeSmallVector<FgString> out;
            out.reserve(systems.size());
            for (const auto se : systems) {
                if (const auto *cs = r.try_get<const CollisionSystem>(se)) out.emplace_back(ToFgStr(cs->Name));
            }
            return out;
        };
        auto cf_view = r.view<const CollisionFilter>();
        for (const auto &[_, e] : ordered_by_source.operator()<SourceCollisionFilterIndex>(cf_view)) {
            const auto &f = cf_view.get<const CollisionFilter>(e);
            collision_filter_to_index[e] = asset.collisionFilters.size();
            fastgltf::CollisionFilter out{.collisionSystems = resolve_system_names(f.Systems), .notCollideWithSystems = {}, .collideWithSystems = {}};
            if (f.Mode == CollideMode::Allowlist) out.collideWithSystems = resolve_system_names(f.CollideSystems);
            else if (f.Mode == CollideMode::Blocklist) out.notCollideWithSystems = resolve_system_names(f.CollideSystems);
            asset.collisionFilters.emplace_back(std::move(out));
        }
    }

    // Empty strings are omitted by fastgltf's writer.
    asset.assetInfo = fastgltf::AssetInfo{
        .gltfVersion = "2.0",
        .minVersion = ToFgStr(sa.MinVersion),
        .copyright = ToFgStr(sa.Copyright),
        .generator = ToFgStr(sa.Generator),
        .extras = ToFgStr(sa.AssetExtras),
        .extensions = ToFgStr(sa.AssetExtensions),
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

    // Strided variant: copy `field` of each element in `data` into `bin` and register a
    // single-attribute accessor — no per-mesh contiguous vector materialized.
    const auto AddFieldAccessor = [&]<typename T, typename V>(std::span<const V> data, T V::*field, fastgltf::AccessorType type, fastgltf::BufferTarget target) {
        const uint32_t off = AppendField<T>(bin, data, field);
        const uint32_t bv = AddBufferView(off, uint32_t(data.size() * sizeof(T)), {}, target);
        return AddAccessor(bv, uint32_t(data.size()), type, fastgltf::ComponentType::Float);
    };
    const auto AddPositionFieldAccessor = [&]<typename V>(std::span<const V> data, vec3 V::*field, fastgltf::BufferTarget target) {
        const uint32_t vcount = uint32_t(data.size());
        if (vcount == 0) return AddFieldAccessor.template operator()<vec3>(data, field, fastgltf::AccessorType::Vec3, target);
        vec3 lo = data[0].*field, hi = lo;
        for (const auto &v : data) {
            lo = glm::min(lo, v.*field);
            hi = glm::max(hi, v.*field);
        }
        const uint32_t off = AppendField<vec3>(bin, data, field);
        const uint32_t bv = AddBufferView(off, vcount * sizeof(vec3), {}, target);
        return AddAccessor(
            bv, vcount, fastgltf::AccessorType::Vec3, fastgltf::ComponentType::Float,
            MakeBounds({lo.x, lo.y, lo.z}), MakeBounds({hi.x, hi.y, hi.z})
        );
    };

    // Emit COLOR_0 from the strided Vertex span, preserving source component count.
    const auto EmitColor0 = [&](fastgltf::pmr::SmallVector<fastgltf::Attribute, 4> &out, std::span<const Vertex> verts, uint8_t component_count) {
        if (component_count == 3) {
            const uint32_t vcount = uint32_t(verts.size());
            const uint32_t off = bin.size();
            bin.resize(off + vcount * sizeof(vec3));
            auto *outp = reinterpret_cast<vec3 *>(bin.data() + off);
            for (uint32_t i = 0; i < vcount; ++i) outp[i] = {verts[i].Color.x, verts[i].Color.y, verts[i].Color.z};
            while (bin.size() % 4 != 0) bin.emplace_back(std::byte{0});
            const uint32_t bv = AddBufferView(off, vcount * sizeof(vec3), {}, fastgltf::BufferTarget::ArrayBuffer);
            out.emplace_back(fastgltf::Attribute{"COLOR_0", AddAccessor(bv, vcount, fastgltf::AccessorType::Vec3, fastgltf::ComponentType::Float)});
        } else {
            out.emplace_back(fastgltf::Attribute{"COLOR_0", AddFieldAccessor.template operator()<vec4>(verts, &Vertex::Color, fastgltf::AccessorType::Vec4, fastgltf::BufferTarget::ArrayBuffer)});
        }
    };

    // Animations: merge per-entity engine clips back into source-side clips by (Name, Duration).
    // Engine splits each source clip into N entity clips (one per affected entity); reverse here.
    // Pre-seed `asset.animations` in source-name order from `gltf::SourceAssets::AnimationOrder` so
    // animations[*] indices match source; clips not in the source list (runtime-added) are appended.
    std::unordered_map<std::string, size_t> clip_index_by_name;
    std::vector<float> clip_duration_by_index;
    if (src_assets) {
        asset.animations.reserve(src_assets->AnimationOrder.size());
        clip_duration_by_index.reserve(src_assets->AnimationOrder.size());
        for (const auto &name : src_assets->AnimationOrder) {
            clip_index_by_name.emplace(name, asset.animations.size());
            asset.animations.emplace_back(fastgltf::Animation{.channels = {}, .samplers = {}, .name = ToFgStr(name)});
            clip_duration_by_index.emplace_back(0.f);
        }
    }
    const auto get_or_create_clip_index = [&](const std::string &name, float duration) -> size_t {
        auto [it, inserted] = clip_index_by_name.try_emplace(name, asset.animations.size());
        if (inserted) {
            asset.animations.emplace_back(fastgltf::Animation{.channels = {}, .samplers = {}, .name = ToFgStr(name)});
            clip_duration_by_index.emplace_back(duration);
        } else {
            clip_duration_by_index[it->second] = std::max(clip_duration_by_index[it->second], duration);
        }
        return it->second;
    };
    // Push one channel into asset.animations[clip_idx]: write times+values accessors, then add a sampler/channel pair.
    const auto push_channel = [&](size_t clip_idx, uint32_t target_node_index, AnimationPath target, AnimationInterpolation interp, std::span<const float> times, std::span<const float> values) {
        if (times.empty()) return;
        const uint32_t t_offset = AppendAligned<float>(bin, times);
        const uint32_t t_bv = AddBufferView(t_offset, uint32_t(times.size() * sizeof(float)));
        const auto [t_min, t_max] = std::minmax_element(times.begin(), times.end());
        const uint32_t t_acc = AddAccessor(
            t_bv, uint32_t(times.size()), fastgltf::AccessorType::Scalar, fastgltf::ComponentType::Float,
            MakeBounds({double(*t_min)}), MakeBounds({double(*t_max)})
        );
        const uint32_t v_offset = AppendAligned<float>(bin, values);
        const uint32_t v_bv = AddBufferView(v_offset, uint32_t(values.size() * sizeof(float)));
        const auto [v_type, v_count] = [&] -> std::pair<fastgltf::AccessorType, uint32_t> {
            switch (target) {
                case AnimationPath::Translation:
                case AnimationPath::Scale: return {fastgltf::AccessorType::Vec3, uint32_t(values.size() / 3)};
                case AnimationPath::Rotation: return {fastgltf::AccessorType::Vec4, uint32_t(values.size() / 4)};
                case AnimationPath::Weights: return {fastgltf::AccessorType::Scalar, uint32_t(values.size())};
            }
        }();
        const uint32_t v_acc = AddAccessor(v_bv, v_count, v_type, fastgltf::ComponentType::Float);
        auto &anim = asset.animations[clip_idx];
        anim.samplers.emplace_back(fastgltf::AnimationSampler{.inputAccessor = t_acc, .outputAccessor = v_acc, .interpolation = FromInterp(interp)});
        anim.channels.emplace_back(fastgltf::AnimationChannel{.samplerIndex = anim.samplers.size() - 1, .nodeIndex = target_node_index, .path = FromPath(target)});
    };
    const auto get_node_index = [&](entt::entity e) -> std::optional<uint32_t> {
        const auto it = entity_to_node_index.find(e);
        return it != entity_to_node_index.end() ? std::optional<uint32_t>{it->second} : std::nullopt;
    };

    // Armature animation: bone channels → joint node index.
    for (const auto [data_entity, anim] : r.view<const ArmatureAnimation>().each()) {
        const auto &arm = r.get<const Armature>(data_entity);
        for (const auto &clip : anim.Clips) {
            const auto idx = get_or_create_clip_index(clip.Name, clip.DurationSeconds);
            for (const auto &ch : clip.Channels) {
                if (ch.BoneIndex == InvalidBoneIndex || ch.BoneIndex >= arm.Bones.size()) continue;
                const auto &bone = arm.Bones[ch.BoneIndex];
                if (!bone.JointNodeIndex) continue;
                push_channel(idx, *bone.JointNodeIndex, ch.Target, ch.Interp, ch.TimesSeconds, ch.Values);
            }
        }
    }
    // Morph weight animation: target = the mesh-instance entity's node index.
    for (const auto [entity, anim] : r.view<const MorphWeightAnimation>().each()) {
        const auto node_idx = get_node_index(entity);
        if (!node_idx) continue;
        for (const auto &clip : anim.Clips) {
            const auto idx = get_or_create_clip_index(clip.Name, clip.DurationSeconds);
            for (const auto &ch : clip.Channels) {
                push_channel(idx, *node_idx, AnimationPath::Weights, ch.Interp, ch.TimesSeconds, ch.Values);
            }
        }
    }
    // Node transform animation: target = the object entity's node index.
    for (const auto [entity, anim] : r.view<const NodeTransformAnimation>().each()) {
        const auto node_idx = get_node_index(entity);
        if (!node_idx) continue;
        for (const auto &clip : anim.Clips) {
            const auto idx = get_or_create_clip_index(clip.Name, clip.DurationSeconds);
            for (const auto &ch : clip.Channels) {
                push_channel(idx, *node_idx, ch.Target, ch.Interp, ch.TimesSeconds, ch.Values);
            }
        }
    }

    // Samplers
    asset.samplers.reserve(sa.Samplers.size());
    for (const auto &s : sa.Samplers) {
        asset.samplers.emplace_back(fastgltf::Sampler{
            .magFilter = ToFgOpt<fastgltf::Filter>(s.MagFilter, FromFilter),
            .minFilter = ToFgOpt<fastgltf::Filter>(s.MinFilter, FromFilter),
            .wrapS = FromWrap(s.WrapS),
            .wrapT = FromWrap(s.WrapT),
            .name = ToFgStr(s.Name),
        });
    }

    // Images: dirty → re-encode from GPU readback; external URI not dirty → re-read source +
    // magic-byte validate, fall back to embed-as-PNG on miss; embedded → passthrough Bytes.
    asset.images.reserve(sa.Images.size());

    std::unordered_map<uint32_t, const TextureEntry *> texture_for_image;
    for (const auto &tex : sc.Textures.Textures) {
        if (tex.SourceImageIndex != UINT32_MAX) texture_for_image.emplace(tex.SourceImageIndex, &tex);
    }
    const auto magic_matches = [](std::span<const std::byte> b, gltf::MimeType mime) {
        const auto *u8 = reinterpret_cast<const uint8_t *>(b.data());
        using enum gltf::MimeType;
        switch (mime) {
            case PNG: return b.size() >= 4 && u8[0] == 0x89 && u8[1] == 0x50 && u8[2] == 0x4E && u8[3] == 0x47;
            case JPEG: return b.size() >= 3 && u8[0] == 0xFF && u8[1] == 0xD8 && u8[2] == 0xFF;
            case WEBP: return b.size() >= 12 && u8[8] == 0x57 && u8[9] == 0x45 && u8[10] == 0x42 && u8[11] == 0x50;
            case KTX2: {
                static constexpr uint8_t Magic[12]{0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A};
                return b.size() >= 12 && std::memcmp(b.data(), Magic, 12) == 0;
            }
            default: return false;
        }
    };
    // Lazy: command pool / fence are created on the first GPU readback only.
    vk::UniqueCommandPool save_pool;
    vk::UniqueFence save_fence;
    const auto reencode_from_gpu = [&](uint32_t img_idx, gltf::MimeType target, std::string_view name)
        -> std::expected<std::pair<std::vector<std::byte>, gltf::MimeType>, std::string> {
        const auto it = texture_for_image.find(img_idx);
        if (it == texture_for_image.end()) return std::unexpected{std::format("Image '{}' has no GPU texture; cannot re-encode.", name)};
        if (!sc.Vk || !sc.BufCtx) return std::unexpected{"GPU readback required but SaveContext.Vk / BufCtx is null"};
        if (!save_pool) {
            save_pool = sc.Vk->Device.createCommandPoolUnique({vk::CommandPoolCreateFlagBits::eTransient, sc.Vk->QueueFamily});
            save_fence = sc.Vk->Device.createFenceUnique({});
        }
        auto rgba8 = ReadbackTextureRgba8(*sc.Vk, *sc.BufCtx, *save_pool, *save_fence, *it->second);
        if (!rgba8) return std::unexpected{std::move(rgba8.error())};
        const auto w = it->second->Width, h = it->second->Height;
        if (auto enc = EncodeImageRgba8ForMime(target, *rgba8, w, h, sc.Options.LossyImageQuality, name)) {
            return std::pair{std::move(*enc), target};
        } else if (target == gltf::MimeType::PNG) {
            return std::unexpected{std::move(enc.error())};
        } else {
            // Unsupported encoder (WebP / KTX2 / DDS): fall back to PNG.
            std::cerr << std::format("Warning: image '{}': {} — falling back to PNG.\n", name, enc.error());
            auto png = EncodeImagePngRgba8(*rgba8, w, h, name);
            if (!png) return std::unexpected{std::move(png.error())};
            return std::pair{std::move(*png), gltf::MimeType::PNG};
        }
    };

    for (uint32_t i = 0; i < sa.Images.size(); ++i) {
        const auto &img = sa.Images[i];
        // Default: passthrough embedded Bytes. Branches below override when re-encoding or when
        // emitting as external URI (which uses no bytes).
        std::vector<std::byte> owned; // backs `view` when we re-encode
        std::span<const std::byte> view = img.Bytes;
        auto emit_mime = img.MimeType;
        bool emit_external_uri = false;
        const bool ktx2_or_dds = img.MimeType == gltf::MimeType::KTX2 || img.MimeType == gltf::MimeType::DDS;
        if (img.IsDirty && !ktx2_or_dds) {
            auto re = reencode_from_gpu(i, img.MimeType, img.Name);
            if (!re) return std::unexpected{std::move(re.error())};
            owned = std::move(re->first);
            emit_mime = re->second;
            view = owned;
        } else if (img.IsDirty) {
            // KTX2/DDS dirty: no encoder dep. Passthrough with a warning.
            std::cerr << std::format("Warning: image '{}' is dirty but {} re-encoding isn't supported; emitting original bytes.\n", img.Name, img.MimeType == gltf::MimeType::KTX2 ? "KTX2" : "DDS");
        } else if (!img.Uri.empty()) {
            std::error_code ec;
            const bool exists = !img.SourceAbsPath.empty() && std::filesystem::is_regular_file(img.SourceAbsPath, ec);
            // Unknown mimes (HDR / IBL panoramas) have no magic check or RGBA8 re-encode path —
            // existence-only.
            const bool validate = img.MimeType == gltf::MimeType::PNG || img.MimeType == gltf::MimeType::JPEG ||
                img.MimeType == gltf::MimeType::WEBP || img.MimeType == gltf::MimeType::KTX2;
            bool ok = false;
            if (exists) {
                if (!validate) ok = true;
                else if (auto b = ReadFileBytes(img.SourceAbsPath)) ok = magic_matches(*b, img.MimeType);
            }
            if (ok) {
                emit_external_uri = true;
            } else if (auto re = reencode_from_gpu(i, gltf::MimeType::PNG, img.Name)) {
                std::cerr << std::format("Warning: image '{}' source '{}' missing or mime-mismatched; embedding as PNG.\n", img.Name, img.SourceAbsPath);
                owned = std::move(re->first);
                emit_mime = gltf::MimeType::PNG;
                view = owned;
            } else {
                // Best-effort: no GPU readback available — emit URI as-is so the save still completes.
                std::cerr << std::format("Warning: image '{}' fallback re-encode failed ({}); emitting URI as-is.\n", img.Name, re.error());
                emit_external_uri = true;
            }
        }

        if (emit_external_uri) {
            const auto fg_mime = img.SourceHadMimeType ? FromMimeType(img.MimeType) : fastgltf::MimeType::None;
            asset.images.emplace_back(fastgltf::Image{
                .data = fastgltf::sources::URI{.fileByteOffset = 0, .uri = fastgltf::URI{std::string_view{img.Uri}}, .mimeType = fg_mime},
                .name = ToFgStr(img.Name),
            });
        } else if (img.SourceDataUri && !view.empty()) {
            const auto fg_mime = FromMimeType(emit_mime);
            const auto mime_str = fg_mime == fastgltf::MimeType::None ? std::string{} : std::string{fastgltf::getMimeTypeString(fg_mime)};
            const auto data_uri = "data:" + mime_str + ";base64," + fastgltf::base64::encode(reinterpret_cast<const std::uint8_t *>(view.data()), view.size());
            asset.images.emplace_back(fastgltf::Image{
                .data = fastgltf::sources::URI{.fileByteOffset = 0, .uri = fastgltf::URI{data_uri}, .mimeType = fastgltf::MimeType::None},
                .name = ToFgStr(img.Name),
            });
        } else {
            uint32_t bv;
            if (!view.empty()) {
                bv = AddBufferView(AppendAligned(bin, view.data(), uint32_t(view.size())), uint32_t(view.size()));
            } else {
                // Placeholder empty bufferView.
                const uint32_t offset = bin.size();
                bin.emplace_back(std::byte{0});
                while (bin.size() % 4 != 0) bin.emplace_back(std::byte{0});
                bv = AddBufferView(offset, 1);
            }
            asset.images.emplace_back(fastgltf::Image{
                .data = fastgltf::sources::BufferView{.bufferViewIndex = bv, .mimeType = FromMimeType(emit_mime)},
                .name = ToFgStr(img.Name),
            });
        }
    }

    // Textures
    asset.textures.reserve(sa.Textures.size());
    for (const auto &t : sa.Textures) {
        asset.textures.emplace_back(fastgltf::Texture{
            .samplerIndex = ToFgOpt<std::size_t>(t.SamplerIndex),
            .imageIndex = ToFgOpt<std::size_t>(t.ImageIndex),
            .basisuImageIndex = ToFgOpt<std::size_t>(t.BasisuImageIndex),
            .ddsImageIndex = ToFgOpt<std::size_t>(t.DdsImageIndex),
            .webpImageIndex = ToFgOpt<std::size_t>(t.WebpImageIndex),
            .name = ToFgStr(t.Name),
        });
    }

    // Materials: skip the engine "Default" at registry index 0 and the trailing synthetic
    // "DefaultMaterial" the loader appends — emit i in [1, material_count - 1). Build
    // fastgltf::Material directly from the GPU PBRMaterial; `MaterialSourceMeta` carries the
    // round-trip-only deltas (ext-block presence, source texture indices, KHR_texture_transform
    // meta, EmissiveFactor*=strength split).
    const uint32_t save_material_count = material_count > 1 ? material_count - 2u : 0u;
    asset.materials.reserve(save_material_count);
    using M = MaterialSourceMeta;
    static const MaterialSourceMeta DefaultMeta{};
    for (uint32_t i = 1; i <= save_material_count; ++i) {
        const auto source_idx = i - 1;
        auto pbr = sc.Buffers.Materials.Get(i);
        const auto &meta = source_idx < material_metas.size() ? material_metas[source_idx] : DefaultMeta;
        const auto bits = meta.ExtensionPresence;
        // Restore source texture slots — PBRMaterial holds bindless slots; emit needs gltf indices.
        pbr.BaseColorTexture.Slot = meta.TextureSlots[MTS_BaseColor];
        pbr.MetallicRoughnessTexture.Slot = meta.TextureSlots[MTS_MetallicRoughness];
        pbr.NormalTexture.Slot = meta.TextureSlots[MTS_Normal];
        pbr.OcclusionTexture.Slot = meta.TextureSlots[MTS_Occlusion];
        pbr.EmissiveTexture.Slot = meta.TextureSlots[MTS_Emissive];
        pbr.Specular.Texture.Slot = meta.TextureSlots[MTS_Specular];
        pbr.Specular.ColorTexture.Slot = meta.TextureSlots[MTS_SpecularColor];
        pbr.Sheen.ColorTexture.Slot = meta.TextureSlots[MTS_SheenColor];
        pbr.Sheen.RoughnessTexture.Slot = meta.TextureSlots[MTS_SheenRoughness];
        pbr.Transmission.Texture.Slot = meta.TextureSlots[MTS_Transmission];
        pbr.DiffuseTransmission.Texture.Slot = meta.TextureSlots[MTS_DiffuseTransmission];
        pbr.DiffuseTransmission.ColorTexture.Slot = meta.TextureSlots[MTS_DiffuseTransmissionColor];
        pbr.Volume.ThicknessTexture.Slot = meta.TextureSlots[MTS_VolumeThickness];
        pbr.Clearcoat.Texture.Slot = meta.TextureSlots[MTS_Clearcoat];
        pbr.Clearcoat.RoughnessTexture.Slot = meta.TextureSlots[MTS_ClearcoatRoughness];
        pbr.Clearcoat.NormalTexture.Slot = meta.TextureSlots[MTS_ClearcoatNormal];
        pbr.Anisotropy.Texture.Slot = meta.TextureSlots[MTS_Anisotropy];
        pbr.Iridescence.Texture.Slot = meta.TextureSlots[MTS_Iridescence];
        pbr.Iridescence.ThicknessTexture.Slot = meta.TextureSlots[MTS_IridescenceThickness];

        std::string name = (!meta.NameWasEmpty && i < names.size()) ? names[i] : std::string{};
        // Un-fold load's `EmissiveFactor *= strength` for emissive_strength round-trip.
        vec3 emissive_factor = pbr.EmissiveFactor;
        if (meta.EmissiveStrength && *meta.EmissiveStrength != 0.f) emissive_factor /= *meta.EmissiveStrength;

        fastgltf::Material out;
        out.name = ToFgStr(name);
        out.pbrData.baseColorFactor = {pbr.BaseColorFactor.x, pbr.BaseColorFactor.y, pbr.BaseColorFactor.z, pbr.BaseColorFactor.w};
        out.pbrData.metallicFactor = pbr.MetallicFactor;
        out.pbrData.roughnessFactor = pbr.RoughnessFactor;
        out.pbrData.baseColorTexture = ToFgTexInfo(pbr.BaseColorTexture, &meta.BaseSlotMeta[0]);
        out.pbrData.metallicRoughnessTexture = ToFgTexInfo(pbr.MetallicRoughnessTexture, &meta.BaseSlotMeta[1]);
        out.normalTexture = ToFgNormalTexInfo(pbr.NormalTexture, pbr.NormalScale, &meta.BaseSlotMeta[2]);
        out.occlusionTexture = ToFgOcclusionTexInfo(pbr.OcclusionTexture, pbr.OcclusionStrength, &meta.BaseSlotMeta[3]);
        out.emissiveTexture = ToFgTexInfo(pbr.EmissiveTexture, &meta.BaseSlotMeta[4]);
        out.emissiveFactor = {emissive_factor.x, emissive_factor.y, emissive_factor.z};
        if (bits & M::ExtEmissiveStrength) out.emissiveStrength = fastgltf::Optional<fastgltf::num>{meta.EmissiveStrength.value_or(1.f)};
        out.alphaMode = FromAlphaMode(pbr.AlphaMode);
        out.alphaCutoff = pbr.AlphaCutoff;
        out.doubleSided = pbr.DoubleSided != 0u;
        out.unlit = pbr.Unlit != 0u;
        if (bits & M::ExtIor) out.ior = fastgltf::Optional<fastgltf::num>{pbr.Ior};
        if (bits & M::ExtDispersion) out.dispersion = fastgltf::Optional<fastgltf::num>{pbr.Dispersion};

        if (bits & M::ExtSheen) {
            out.sheen = std::make_unique<fastgltf::MaterialSheen>();
            out.sheen->sheenColorFactor = {pbr.Sheen.ColorFactor.x, pbr.Sheen.ColorFactor.y, pbr.Sheen.ColorFactor.z};
            out.sheen->sheenRoughnessFactor = pbr.Sheen.RoughnessFactor;
            out.sheen->sheenColorTexture = ToFgTexInfo(pbr.Sheen.ColorTexture);
            out.sheen->sheenRoughnessTexture = ToFgTexInfo(pbr.Sheen.RoughnessTexture);
        }
        if (bits & M::ExtSpecular) {
            out.specular = std::make_unique<fastgltf::MaterialSpecular>();
            out.specular->specularFactor = pbr.Specular.Factor;
            out.specular->specularColorFactor = {pbr.Specular.ColorFactor.x, pbr.Specular.ColorFactor.y, pbr.Specular.ColorFactor.z};
            out.specular->specularTexture = ToFgTexInfo(pbr.Specular.Texture);
            out.specular->specularColorTexture = ToFgTexInfo(pbr.Specular.ColorTexture);
        }
        if (bits & M::ExtTransmission) {
            out.transmission = std::make_unique<fastgltf::MaterialTransmission>();
            out.transmission->transmissionFactor = pbr.Transmission.Factor;
            out.transmission->transmissionTexture = ToFgTexInfo(pbr.Transmission.Texture);
        }
        if (bits & M::ExtDiffuseTransmission) {
            out.diffuseTransmission = std::make_unique<fastgltf::MaterialDiffuseTransmission>();
            out.diffuseTransmission->diffuseTransmissionFactor = pbr.DiffuseTransmission.Factor;
            out.diffuseTransmission->diffuseTransmissionColorFactor = {pbr.DiffuseTransmission.ColorFactor.x, pbr.DiffuseTransmission.ColorFactor.y, pbr.DiffuseTransmission.ColorFactor.z};
            out.diffuseTransmission->diffuseTransmissionTexture = ToFgTexInfo(pbr.DiffuseTransmission.Texture);
            out.diffuseTransmission->diffuseTransmissionColorTexture = ToFgTexInfo(pbr.DiffuseTransmission.ColorTexture);
        }
        if (bits & M::ExtVolume) {
            out.volume = std::make_unique<fastgltf::MaterialVolume>();
            out.volume->thicknessFactor = pbr.Volume.ThicknessFactor;
            out.volume->attenuationColor = {pbr.Volume.AttenuationColor.x, pbr.Volume.AttenuationColor.y, pbr.Volume.AttenuationColor.z};
            out.volume->attenuationDistance = pbr.Volume.AttenuationDistance > 0.f ? pbr.Volume.AttenuationDistance : std::numeric_limits<float>::infinity();
            out.volume->thicknessTexture = ToFgTexInfo(pbr.Volume.ThicknessTexture);
        }
        if (bits & M::ExtClearcoat) {
            out.clearcoat = std::make_unique<fastgltf::MaterialClearcoat>();
            out.clearcoat->clearcoatFactor = pbr.Clearcoat.Factor;
            out.clearcoat->clearcoatRoughnessFactor = pbr.Clearcoat.RoughnessFactor;
            out.clearcoat->clearcoatTexture = ToFgTexInfo(pbr.Clearcoat.Texture);
            out.clearcoat->clearcoatRoughnessTexture = ToFgTexInfo(pbr.Clearcoat.RoughnessTexture);
            out.clearcoat->clearcoatNormalTexture = ToFgNormalTexInfo(pbr.Clearcoat.NormalTexture, pbr.Clearcoat.NormalScale);
        }
        if (bits & M::ExtAnisotropy) {
            out.anisotropy = std::make_unique<fastgltf::MaterialAnisotropy>();
            out.anisotropy->anisotropyStrength = pbr.Anisotropy.Strength;
            out.anisotropy->anisotropyRotation = pbr.Anisotropy.Rotation;
            out.anisotropy->anisotropyTexture = ToFgTexInfo(pbr.Anisotropy.Texture);
        }
        if (bits & M::ExtIridescence) {
            out.iridescence = std::make_unique<fastgltf::MaterialIridescence>();
            out.iridescence->iridescenceFactor = pbr.Iridescence.Factor;
            out.iridescence->iridescenceIor = pbr.Iridescence.Ior;
            out.iridescence->iridescenceThicknessMinimum = pbr.Iridescence.ThicknessMinimum;
            out.iridescence->iridescenceThicknessMaximum = pbr.Iridescence.ThicknessMaximum;
            out.iridescence->iridescenceTexture = ToFgTexInfo(pbr.Iridescence.Texture);
            out.iridescence->iridescenceThicknessTexture = ToFgTexInfo(pbr.Iridescence.ThicknessTexture);
        }

        asset.materials.emplace_back(std::move(out));
    }

    asset.meshes.reserve(mesh_groups.size());
    // For non-triangle (Lines/Points) primitives: emit NORMAL/COLOR_0 iff at least one vertex
    // has a non-default value (sentinel = NORMAL=0, COLOR_0=(1,1,1,1)). CPU stores vec4 colors.
    const auto emit_non_triangle_attrs = [&](fastgltf::pmr::SmallVector<fastgltf::Attribute, 4> &out, std::span<const Vertex> verts) {
        if (std::ranges::any_of(verts, [](const auto &v) { return v.Normal != vec3{0}; })) {
            out.emplace_back(fastgltf::Attribute{"NORMAL", AddFieldAccessor.template operator()<vec3>(verts, &Vertex::Normal, fastgltf::AccessorType::Vec3, fastgltf::BufferTarget::ArrayBuffer)});
        }
        if (std::ranges::any_of(verts, [](const auto &v) { return v.Color != vec4{1}; })) {
            EmitColor0(out, verts, /*component_count=*/4);
        }
    };
    for (uint32_t mi = 0; mi < mesh_groups.size(); ++mi) {
        const auto &group = mesh_groups[mi];
        fastgltf::pmr::MaybeSmallVector<fastgltf::Primitive, 2> primitives;
        fastgltf::pmr::MaybeSmallVector<fastgltf::num> default_weights;

        // --- Triangle primitives ---
        // Each source primitive owns a contiguous [offset, offset+count) slice of the merged
        // per-mesh vertex buffer; re-slice here by primitive range. Vertex/face/skin/morph data
        // is read straight from MeshStore — no per-mesh contiguous vectors materialized.
        if (group.Triangles != entt::null) {
            const auto &mesh = r.get<const Mesh>(group.Triangles);
            const auto &layout = r.get<const MeshSourceLayout>(group.Triangles);
            const auto store_id = mesh.GetStoreId();
            const auto vertices = meshes.GetVertices(store_id);
            const auto total_vcount = uint32_t(vertices.size());
            const auto face_primitives = meshes.GetFacePrimitiveIndices(store_id);
            const auto primitive_materials = meshes.GetPrimitiveMaterialIndices(store_id);
            const auto prim_count = uint32_t(layout.VertexCounts.size());

            std::vector<uint32_t> vertex_offsets(prim_count + 1, 0);
            for (uint32_t p = 0; p < prim_count; ++p) vertex_offsets[p + 1] = vertex_offsets[p] + layout.VertexCounts[p];

            // Build per-primitive (rebased, fan-triangulated) index buffers in one faces walk.
            std::vector<std::vector<uint32_t>> indices_per_prim(prim_count);
            uint32_t fi = 0;
            for (const auto fh : mesh.faces()) {
                const uint32_t p = fi < face_primitives.size() ? face_primitives[fi] : 0u;
                if (p < prim_count) {
                    std::array<uint32_t, 16> fv{};
                    uint32_t fv_count = 0;
                    for (const auto vh : mesh.fv_range(fh)) {
                        if (fv_count < fv.size()) fv[fv_count] = *vh;
                        ++fv_count;
                    }
                    if (fv_count >= 3) {
                        const auto offset = vertex_offsets[p];
                        auto &out = indices_per_prim[p];
                        for (uint32_t k = 1; k + 1 < fv_count; ++k) {
                            out.emplace_back(fv[0] - offset);
                            out.emplace_back(fv[k] - offset);
                            out.emplace_back(fv[k + 1] - offset);
                        }
                    }
                }
                ++fi;
            }

            // Skin / morph spans (empty when the mesh lacks the channel).
            const auto bd_range = meshes.GetBoneDeformRange(store_id);
            const auto bd_span = bd_range.Count > 0 ? meshes.BoneDeformBuffer.Get(bd_range) : std::span<const BoneDeformVertex>{};
            const bool has_skin = bd_span.size() == total_vcount && total_vcount > 0;
            const uint32_t target_count = (total_vcount > 0) ? meshes.GetMorphTargetCount(store_id) : 0u;
            const auto mt_span = target_count > 0 ? meshes.MorphTargetBuffer.Get(meshes.GetMorphTargetRange(store_id)) : std::span<const MorphTargetVertex>{};
            // CreateMesh writes 0 when source lacked normal deltas, so any non-zero ⇒ source had them.
            const bool has_normal_deltas = std::ranges::any_of(mt_span, [](const auto &m) { return m.NormalDelta != vec3{0}; });
            const bool has_tangent_deltas = !layout.MorphTangentDeltas.empty();

            // OR of per-prim AttributeFlags; channel emitted iff any primitive set the bit
            // (matches old behavior of populating the merged TriangleAttrs slot).
            uint32_t any_flags = 0;
            for (const auto f : layout.AttributeFlags) any_flags |= f;
            const bool have_flags = layout.AttributeFlags.size() == prim_count;

            for (uint32_t prim_idx = 0; prim_idx < prim_count; ++prim_idx) {
                const uint32_t pcount = layout.VertexCounts[prim_idx];
                if (pcount == 0) continue; // non-triangle primitive
                const auto offset = vertex_offsets[prim_idx];
                const auto flags = have_flags ? layout.AttributeFlags[prim_idx] : ~0u;
                const auto vslice = vertices.subspan(offset, pcount);

                fastgltf::pmr::SmallVector<fastgltf::Attribute, 4> prim_attrs;
                prim_attrs.emplace_back(fastgltf::Attribute{"POSITION", AddPositionFieldAccessor.template operator()<Vertex>(vslice, &Vertex::Position, fastgltf::BufferTarget::ArrayBuffer)});

                const auto emit = [&]<typename T>(const char *name, T Vertex::*field, fastgltf::AccessorType type) {
                    prim_attrs.emplace_back(fastgltf::Attribute{name, AddFieldAccessor.template operator()<T>(vslice, field, type, fastgltf::BufferTarget::ArrayBuffer)});
                };
                if ((any_flags & MeshAttributeBit_Normal) && (flags & MeshAttributeBit_Normal)) emit("NORMAL", &Vertex::Normal, fastgltf::AccessorType::Vec3);
                if ((any_flags & MeshAttributeBit_Tangent) && (flags & MeshAttributeBit_Tangent)) emit("TANGENT", &Vertex::Tangent, fastgltf::AccessorType::Vec4);
                if ((any_flags & MeshAttributeBit_Color0) && (flags & MeshAttributeBit_Color0)) EmitColor0(prim_attrs, vslice, layout.Colors0ComponentCount);
                if ((any_flags & MeshAttributeBit_TexCoord0) && (flags & MeshAttributeBit_TexCoord0)) emit("TEXCOORD_0", &Vertex::TexCoord0, fastgltf::AccessorType::Vec2);
                if ((any_flags & MeshAttributeBit_TexCoord1) && (flags & MeshAttributeBit_TexCoord1)) emit("TEXCOORD_1", &Vertex::TexCoord1, fastgltf::AccessorType::Vec2);
                if ((any_flags & MeshAttributeBit_TexCoord2) && (flags & MeshAttributeBit_TexCoord2)) emit("TEXCOORD_2", &Vertex::TexCoord2, fastgltf::AccessorType::Vec2);
                if ((any_flags & MeshAttributeBit_TexCoord3) && (flags & MeshAttributeBit_TexCoord3)) emit("TEXCOORD_3", &Vertex::TexCoord3, fastgltf::AccessorType::Vec2);

                if (has_skin) {
                    const auto bd_slice = bd_span.subspan(offset, pcount);
                    // Convert uvec4 → uint16_t[4] strided directly into bin (no intermediate vector).
                    const uint32_t j_off = bin.size();
                    bin.resize(j_off + pcount * 4 * sizeof(uint16_t));
                    auto *jp = reinterpret_cast<uint16_t *>(bin.data() + j_off);
                    for (uint32_t i = 0; i < pcount; ++i) {
                        const auto &j = bd_slice[i].Joints;
                        jp[i * 4 + 0] = uint16_t(j.x);
                        jp[i * 4 + 1] = uint16_t(j.y);
                        jp[i * 4 + 2] = uint16_t(j.z);
                        jp[i * 4 + 3] = uint16_t(j.w);
                    }
                    while (bin.size() % 4 != 0) bin.emplace_back(std::byte{0});
                    const uint32_t j_bv = AddBufferView(j_off, pcount * 4 * sizeof(uint16_t), {}, fastgltf::BufferTarget::ArrayBuffer);
                    prim_attrs.emplace_back(fastgltf::Attribute{"JOINTS_0", AddAccessor(j_bv, pcount, fastgltf::AccessorType::Vec4, fastgltf::ComponentType::UnsignedShort)});
                    prim_attrs.emplace_back(fastgltf::Attribute{"WEIGHTS_0", AddFieldAccessor.template operator()<vec4>(bd_slice, &BoneDeformVertex::Weights, fastgltf::AccessorType::Vec4, fastgltf::BufferTarget::ArrayBuffer)});
                }

                std::pmr::vector<fastgltf::pmr::SmallVector<fastgltf::Attribute, 4>> prim_targets;
                if (target_count > 0) {
                    prim_targets.reserve(target_count);
                    for (uint32_t t = 0; t < target_count; ++t) {
                        fastgltf::pmr::SmallVector<fastgltf::Attribute, 4> tattrs;
                        const auto target_slice = mt_span.subspan(std::size_t(t) * total_vcount + offset, pcount);
                        tattrs.emplace_back(fastgltf::Attribute{"POSITION", AddFieldAccessor.template operator()<vec3>(target_slice, &MorphTargetVertex::PositionDelta, fastgltf::AccessorType::Vec3, fastgltf::BufferTarget::ArrayBuffer)});
                        if (has_normal_deltas) tattrs.emplace_back(fastgltf::Attribute{"NORMAL", AddFieldAccessor.template operator()<vec3>(target_slice, &MorphTargetVertex::NormalDelta, fastgltf::AccessorType::Vec3, fastgltf::BufferTarget::ArrayBuffer)});
                        if (has_tangent_deltas) {
                            const std::span<const vec3> tan_deltas(layout.MorphTangentDeltas.data() + std::size_t(t) * total_vcount + offset, pcount);
                            tattrs.emplace_back(fastgltf::Attribute{"TANGENT", AddVec3Accessor(tan_deltas, false, fastgltf::BufferTarget::ArrayBuffer)});
                        }
                        prim_targets.emplace_back(std::move(tattrs));
                    }
                }

                if (indices_per_prim[prim_idx].empty()) continue;
                // Empty HasSourceIndices falls back to emitting (preserves legacy behavior).
                const bool emit_indices = prim_idx < layout.HasSourceIndices.size() ? layout.HasSourceIndices[prim_idx] != 0 : true;
                fastgltf::Optional<std::size_t> indices_accessor;
                if (emit_indices) {
                    indices_accessor = AddDataAccessor(std::span<const uint32_t>(indices_per_prim[prim_idx]), fastgltf::AccessorType::Scalar, fastgltf::ComponentType::UnsignedInt, fastgltf::BufferTarget::ElementArrayBuffer);
                }

                fastgltf::Optional<std::size_t> material_index;
                if (prim_idx < primitive_materials.size()) {
                    // Reverse populate's +1 material-index shift; `~0u` (registry default) = don't emit.
                    const auto reg_idx = primitive_materials[prim_idx];
                    const auto mat = reg_idx >= 1 ? reg_idx - 1 : ~0u;
                    if (mat < save_material_count) material_index = mat;
                }

                std::vector<fastgltf::Optional<std::size_t>> mappings;
                if (prim_idx < layout.VariantMappings.size()) {
                    for (const auto &m : layout.VariantMappings[prim_idx]) {
                        if (m.has_value() && *m < save_material_count) mappings.emplace_back(std::size_t(*m));
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

            const auto dw = meshes.GetDefaultMorphWeights(store_id);
            if (!dw.empty()) {
                default_weights.reserve(dw.size());
                for (const auto w : dw) default_weights.emplace_back(w);
            }
        }

        // --- Line primitive ---
        if (group.Lines != entt::null) {
            const auto &mesh = r.get<const Mesh>(group.Lines);
            const auto vertices = meshes.GetVertices(mesh.GetStoreId());
            if (!vertices.empty() && mesh.EdgeCount() > 0) {
                std::vector<uint32_t> idx;
                idx.reserve(mesh.EdgeCount() * 2);
                for (const auto eh : mesh.edges()) {
                    const auto h0 = mesh.GetHalfedge(eh, 0);
                    idx.emplace_back(*mesh.GetFromVertex(h0));
                    idx.emplace_back(*mesh.GetToVertex(h0));
                }
                fastgltf::pmr::SmallVector<fastgltf::Attribute, 4> attrs;
                attrs.emplace_back(fastgltf::Attribute{"POSITION", AddPositionFieldAccessor.template operator()<Vertex>(vertices, &Vertex::Position, fastgltf::BufferTarget::ArrayBuffer)});
                emit_non_triangle_attrs(attrs, vertices);
                primitives.emplace_back(fastgltf::Primitive{
                    .attributes = std::move(attrs),
                    .type = fastgltf::PrimitiveType::Lines,
                    .targets = {},
                    .indicesAccessor = AddDataAccessor(std::span<const uint32_t>(idx), fastgltf::AccessorType::Scalar, fastgltf::ComponentType::UnsignedInt, fastgltf::BufferTarget::ElementArrayBuffer),
                    .materialIndex = {},
                    .mappings = {},
                    .dracoCompression = nullptr,
                });
            }
        }

        // --- Point primitive ---
        if (group.Points != entt::null) {
            const auto &mesh = r.get<const Mesh>(group.Points);
            const auto vertices = meshes.GetVertices(mesh.GetStoreId());
            if (!vertices.empty()) {
                fastgltf::pmr::SmallVector<fastgltf::Attribute, 4> attrs;
                attrs.emplace_back(fastgltf::Attribute{"POSITION", AddPositionFieldAccessor.template operator()<Vertex>(vertices, &Vertex::Position, fastgltf::BufferTarget::ArrayBuffer)});
                emit_non_triangle_attrs(attrs, vertices);
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

        asset.meshes.emplace_back(fastgltf::Mesh{
            .primitives = std::move(primitives),
            .weights = std::move(default_weights),
            .name = ToFgStr(group.Name),
        });
    }

    // Source→dense skin index remap for node refs below. Per Armature, emit one fastgltf::Skin
    // directly into asset.skins (only joint indices, IBMs, skeleton, and name are needed at the
    // glTF level; runtime-only fields like rest-local transforms and bone names live in the
    // engine's `Armature` and round-trip through node entries instead). Load creates Armature
    // entities in source-skin-index order, so reverse-iterating the view (entt iterates in
    // reverse-creation order) yields source-ascending order without an extra sort.
    std::unordered_map<uint32_t, uint32_t> skin_remap;
    const auto armature_view = r.view<const Armature>();
    for (auto it = armature_view.rbegin(); it != armature_view.rend(); ++it) {
        const auto data_entity = *it;
        const auto &arm = r.get<const Armature>(data_entity);
        if (!arm.ImportedSkin) continue;
        const auto source_skin_index = arm.ImportedSkin->SkinIndex;

        fastgltf::pmr::MaybeSmallVector<std::size_t> joints;
        joints.reserve(arm.ImportedSkin->OrderedJointNodeIndices.size());
        for (const auto j : arm.ImportedSkin->OrderedJointNodeIndices) joints.emplace_back(j);

        const auto ibm = [&]() -> fastgltf::Optional<std::size_t> {
            if (arm.ImportedSkin->InverseBindMatrices.empty()) return {};
            return AddDataAccessor(std::span<const mat4>(arm.ImportedSkin->InverseBindMatrices), fastgltf::AccessorType::Mat4, fastgltf::ComponentType::Float);
        }();

        std::string skin_name;
        if (const auto ait = armature_data_to_object.find(data_entity); ait != armature_data_to_object.end()) {
            if (const auto *sn = r.try_get<const SkinName>(ait->second)) skin_name = sn->Value;
            else if (!r.all_of<SourceEmptyName>(ait->second)) {
                if (const auto *name = r.try_get<const Name>(ait->second)) skin_name = name->Value;
            }
        }

        skin_remap[source_skin_index] = asset.skins.size();
        asset.skins.emplace_back(fastgltf::Skin{
            .inverseBindMatrices = ibm,
            .skeleton = ToFgOpt<std::size_t>(arm.ImportedSkin->SkeletonNodeIndex),
            .joints = std::move(joints),
            .name = ToFgStr(skin_name),
        });
    }

    // Cameras / lights: emit in the order gathered above.
    asset.cameras.reserve(camera_entities_ordered.size());
    for (const auto entity : camera_entities_ordered) {
        const auto *cn = r.try_get<const CameraName>(entity);
        asset.cameras.emplace_back(ConvertCameraToFg(r.get<const ::Camera>(entity), cn ? cn->Value : std::string_view{}));
    }
    asset.lights.reserve(light_entities_ordered.size());
    for (const auto entity : light_entities_ordered) {
        const auto *ln = r.try_get<const LightName>(entity);
        asset.lights.emplace_back(ConvertLightToFg(r.get<const PunctualLight>(entity), ln ? ln->Value : std::string_view{}));
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

    // Nodes — each fastgltf::Node is emitted directly from per-entity registry components,
    // no parallel staging struct. `entity == null` slots fall through with default fields
    // (gaps in the source SourceNodeIndex sequence).
    asset.nodes.reserve(total_node_count);
    bool uses_gpu_instancing = false;
    bool uses_physics_rigid_bodies = false;
    for (uint32_t ni = 0; ni < total_node_count; ++ni) {
        const auto entity = node_to_entity[ni];

        fastgltf::pmr::MaybeSmallVector<std::size_t> children;
        if (const auto cit = children_by_parent.find(ni); cit != children_by_parent.end()) {
            children.reserve(cit->second.size());
            for (const auto &[_, child_idx] : cit->second) children.emplace_back(child_idx);
        }

        if (entity == entt::null) {
            fastgltf::Node empty{};
            empty.children = std::move(children);
            asset.nodes.emplace_back(std::move(empty));
            continue;
        }

        // Refs
        fastgltf::Optional<std::size_t> mesh_index, camera_index, light_index, skin_index;
        if (const auto *inst = r.try_get<const Instance>(entity)) {
            if (const auto it = mesh_entity_to_index.find(inst->Entity); it != mesh_entity_to_index.end()) mesh_index = it->second;
        }
        if (const auto cit = camera_entity_to_index.find(entity); cit != camera_entity_to_index.end()) camera_index = cit->second;
        if (const auto lit = light_entity_to_index.find(entity); lit != light_entity_to_index.end()) light_index = lit->second;
        if (const auto *am = r.try_get<const ArmatureModifier>(entity)) {
            if (const auto it = armature_data_to_skin_index.find(am->ArmatureEntity); it != armature_data_to_skin_index.end()) {
                if (const auto sit = skin_remap.find(it->second); sit != skin_remap.end()) skin_index = sit->second;
            }
        }

        // Name
        std::string node_name;
        if (!r.all_of<SourceEmptyName>(entity)) {
            if (const auto *son = r.try_get<const SourceObjectName>(entity)) node_name = son->Value;
            else if (const auto *nm = r.try_get<const Name>(entity)) node_name = nm->Value;
        }

        const auto &local_transform = r.get<const Transform>(entity);
        const auto &world_transform = r.get<const WorldTransform>(entity);

        // EXT_mesh_gpu_instancing. Per-instance TRS = node.WorldTransform^-1 * instance.WorldTransform.
        // Emit only channels that aren't uniformly default; spec requires >=1 attribute so fall back
        // to TRANSLATION when everything's default.
        const auto &instance_worlds = node_instance_worlds[ni];
        const bool needs_instancing = mesh_index.has_value() && instance_worlds.size() > 1;
        if (needs_instancing) uses_gpu_instancing = true;
        std::pmr::vector<fastgltf::Attribute> instancing;
        if (needs_instancing) {
            const uint32_t count = uint32_t(instance_worlds.size());
            const mat4 node_world_inv = glm::inverse(ToMatrix(world_transform));

            std::vector<vec3> translations(count);
            std::vector<vec4> rotations(count); // xyzw
            std::vector<vec3> scales(count);
            bool any_t = false, any_r = false, any_s = false;
            for (uint32_t i = 0; i < count; ++i) {
                const Transform local = ToTransform(node_world_inv * ToMatrix(instance_worlds[i]));
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

        // Physics
        const auto *motion = r.try_get<const PhysicsMotion>(entity);
        const auto *velocity = r.try_get<const PhysicsVelocity>(entity);
        const auto *cs = r.try_get<const ColliderShape>(entity);
        const auto *tn = r.try_get<const TriggerNodes>(entity);
        const auto *pj = r.try_get<const PhysicsJoint>(entity);
        const bool is_trigger = r.all_of<TriggerTag>(entity);
        fastgltf::Optional<std::size_t> collider_mesh_idx;
        if (cs && cs->MeshEntity != null_entity) {
            if (const auto mit = mesh_entity_to_index.find(cs->MeshEntity); mit != mesh_entity_to_index.end()) collider_mesh_idx = mit->second;
        }

        std::unique_ptr<fastgltf::PhysicsRigidBody> physics_rigid_body;
        if (motion || cs || tn || pj) {
            physics_rigid_body = std::make_unique<fastgltf::PhysicsRigidBody>();
            uses_physics_rigid_bodies = true;

            if (motion) {
                fastgltf::Motion fg_motion{};
                fg_motion.isKinematic = motion->IsKinematic;
                if (motion->Mass) fg_motion.mass = fastgltf::Optional<fastgltf::num>{*motion->Mass};
                if (motion->CenterOfMass) fg_motion.centerOfMass = {motion->CenterOfMass->x, motion->CenterOfMass->y, motion->CenterOfMass->z};
                if (motion->InertiaDiagonal) {
                    fg_motion.inertialDiagonal = fastgltf::Optional<fastgltf::math::fvec3>{fastgltf::math::fvec3{motion->InertiaDiagonal->x, motion->InertiaDiagonal->y, motion->InertiaDiagonal->z}};
                }
                if (motion->InertiaOrientation) {
                    fg_motion.inertialOrientation = fastgltf::Optional<fastgltf::math::fvec4>{fastgltf::math::fvec4{motion->InertiaOrientation->x, motion->InertiaOrientation->y, motion->InertiaOrientation->z, motion->InertiaOrientation->w}};
                }
                fg_motion.gravityFactor = motion->GravityFactor;
                if (velocity) {
                    fg_motion.linearVelocity = {velocity->Linear.x, velocity->Linear.y, velocity->Linear.z};
                    fg_motion.angularVelocity = {velocity->Angular.x, velocity->Angular.y, velocity->Angular.z};
                }
                physics_rigid_body->motion = fastgltf::Optional<fastgltf::Motion>{std::move(fg_motion)};
            }

            if (cs && !is_trigger) {
                fastgltf::Collider collider{};
                if (auto fg_shape = to_fg_shape(cs->Shape)) {
                    collider.geometry.shape = emit_shape_index(*fg_shape);
                } else if (std::holds_alternative<physics::ConvexHull>(cs->Shape)) {
                    if (collider_mesh_idx) collider.geometry.mesh = *collider_mesh_idx;
                    collider.geometry.convexHull = true;
                } else if (std::holds_alternative<physics::TriangleMesh>(cs->Shape)) {
                    if (collider_mesh_idx) collider.geometry.mesh = *collider_mesh_idx;
                }
                if (const auto *cm = r.try_get<const ColliderMaterial>(entity)) {
                    if (const auto mit = physics_material_to_index.find(cm->PhysicsMaterialEntity); mit != physics_material_to_index.end()) collider.physicsMaterial = mit->second;
                    if (const auto fit = collision_filter_to_index.find(cm->CollisionFilterEntity); fit != collision_filter_to_index.end()) collider.collisionFilter = fit->second;
                }
                physics_rigid_body->collider = fastgltf::Optional<fastgltf::Collider>{std::move(collider)};
            }

            using TriggerVariant = std::variant<fastgltf::GeometryTrigger, fastgltf::NodeTrigger>;
            if (cs && is_trigger) {
                // GeometryTrigger: shape on the same entity, distinguished by TriggerTag.
                fastgltf::GeometryTrigger gt{};
                if (auto fg_shape = to_fg_shape(cs->Shape)) {
                    gt.geometry.shape = emit_shape_index(*fg_shape);
                } else if (collider_mesh_idx) {
                    gt.geometry.mesh = *collider_mesh_idx;
                    gt.geometry.convexHull = std::holds_alternative<physics::ConvexHull>(cs->Shape);
                }
                if (const auto *cm = r.try_get<const ColliderMaterial>(entity)) {
                    if (const auto fit = collision_filter_to_index.find(cm->CollisionFilterEntity); fit != collision_filter_to_index.end()) gt.collisionFilter = fit->second;
                }
                physics_rigid_body->trigger = fastgltf::Optional<TriggerVariant>{TriggerVariant{std::move(gt)}};
            } else if (tn && !tn->Nodes.empty()) {
                // NodesTrigger: compound zone.
                fastgltf::NodeTrigger nt;
                for (const auto ne : tn->Nodes) {
                    if (const auto nit = entity_to_node_index.find(ne); nit != entity_to_node_index.end()) nt.nodes.emplace_back(nit->second);
                }
                physics_rigid_body->trigger = fastgltf::Optional<TriggerVariant>{TriggerVariant{std::move(nt)}};
            }

            if (pj) {
                fastgltf::Joint joint{};
                if (const auto cit = entity_to_node_index.find(pj->ConnectedNode); cit != entity_to_node_index.end()) joint.connectedNode = cit->second;
                if (const auto dit = physics_jointdef_to_index.find(pj->JointDefEntity); dit != physics_jointdef_to_index.end()) joint.joint = dit->second;
                joint.enableCollision = pj->EnableCollision;
                physics_rigid_body->joint = fastgltf::Optional<fastgltf::Joint>{std::move(joint)};
            }
        }

        // Matrix-form source round-trips as matrix; TRS-form emits from LocalTransform.
        const auto *source_matrix = r.try_get<const SourceMatrixTransform>(entity);
        const auto fg_transform = [&]() -> std::variant<fastgltf::TRS, fastgltf::math::fmat4x4> {
            if (source_matrix) {
                fastgltf::math::fmat4x4 out;
                for (std::size_t c = 0; c < 4; ++c) {
                    for (std::size_t r2 = 0; r2 < 4; ++r2) out[c][r2] = source_matrix->Value[c][r2];
                }
                return out;
            }
            return fastgltf::TRS{
                .translation = {local_transform.P.x, local_transform.P.y, local_transform.P.z},
                .rotation = fastgltf::math::fquat(local_transform.R.x, local_transform.R.y, local_transform.R.z, local_transform.R.w),
                .scale = {local_transform.S.x, local_transform.S.y, local_transform.S.z},
            };
        }();

        asset.nodes.emplace_back(fastgltf::Node{
            .meshIndex = mesh_index,
            .skinIndex = skin_index,
            .cameraIndex = camera_index,
            .lightIndex = light_index,
            .children = std::move(children),
            .weights = {},
            .transform = fg_transform,
            .instancingAttributes = std::move(instancing),
            .name = ToFgStr(node_name),
            .physicsRigidBody = std::move(physics_rigid_body),
            .visible = true,
            .selectable = true,
            .hoverable = true,
        });
    }

    fastgltf::pmr::MaybeSmallVector<std::size_t> scene_roots;
    if (!sa.DefaultSceneRoots.empty()) {
        scene_roots.reserve(sa.DefaultSceneRoots.size());
        for (const auto n : sa.DefaultSceneRoots) scene_roots.emplace_back(n);
    } else {
        for (uint32_t ni = 0; ni < total_node_count; ++ni) {
            const auto entity = node_to_entity[ni];
            if (entity == entt::null) continue;
            if (const auto *spi = r.try_get<const SourceParentNodeIndex>(entity);
                spi && spi->Value < total_node_count && node_to_entity[spi->Value] != entt::null) continue;
            scene_roots.emplace_back(ni);
        }
    }
    // EXT_lights_image_based
    fastgltf::Optional<std::size_t> scene_ibl_index;
    if (sa.ImageBasedLight) {
        const auto &src_ibl = *sa.ImageBasedLight;
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
        .name = ToFgStr(sa.DefaultSceneName),
    });
    asset.defaultScene = 0;

    asset.extensionsRequired.reserve(sa.ExtensionsRequired.size());
    for (const auto &e : sa.ExtensionsRequired) asset.extensionsRequired.emplace_back(e);

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
    if (!sa.MaterialVariants.empty()) {
        asset.materialVariants.reserve(sa.MaterialVariants.size());
        for (const auto &v : sa.MaterialVariants) asset.materialVariants.emplace_back(v);
        asset.extensionsUsed.emplace_back("KHR_materials_variants");
    }
    // KHR_texture_transform: emitted per-texture when non-identity or when source had it at all.
    // Each emit site (`ToFgTexInfo` / `ToFgNormalTexInfo` / `ToFgOcclusionTexInfo`) attaches the
    // unique_ptr<TextureTransform> exactly when those conditions hold, so checking it here matches.
    {
        const auto has_xf = [](const auto &opt) { return opt.has_value() && opt->transform != nullptr; };
        const auto material_has_xf = [&](const fastgltf::Material &m) {
            if (has_xf(m.pbrData.baseColorTexture) || has_xf(m.pbrData.metallicRoughnessTexture) ||
                has_xf(m.normalTexture) || has_xf(m.occlusionTexture) || has_xf(m.emissiveTexture)) return true;
            if (m.sheen && (has_xf(m.sheen->sheenColorTexture) || has_xf(m.sheen->sheenRoughnessTexture))) return true;
            if (m.specular && (has_xf(m.specular->specularTexture) || has_xf(m.specular->specularColorTexture))) return true;
            if (m.transmission && has_xf(m.transmission->transmissionTexture)) return true;
            if (m.diffuseTransmission && (has_xf(m.diffuseTransmission->diffuseTransmissionTexture) || has_xf(m.diffuseTransmission->diffuseTransmissionColorTexture))) return true;
            if (m.volume && has_xf(m.volume->thicknessTexture)) return true;
            if (m.clearcoat && (has_xf(m.clearcoat->clearcoatTexture) || has_xf(m.clearcoat->clearcoatRoughnessTexture) || has_xf(m.clearcoat->clearcoatNormalTexture))) return true;
            if (m.anisotropy && has_xf(m.anisotropy->anisotropyTexture)) return true;
            if (m.iridescence && (has_xf(m.iridescence->iridescenceTexture) || has_xf(m.iridescence->iridescenceThicknessTexture))) return true;
            return false;
        };
        if (std::ranges::any_of(asset.materials, material_has_xf)) asset.extensionsUsed.emplace_back("KHR_texture_transform");
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
    if (!sa.ExtrasByEntity.empty()) {
        exporter.setUserPointer(const_cast<ExtrasMap *>(&sa.ExtrasByEntity));
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
