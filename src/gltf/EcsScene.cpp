#include "EcsScene.h"

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

// Internal-only intermediate carriers (TU-local). MeshData specifically holds parsed-but-not-yet-
// uploaded geometry: load builds it from fastgltf accessors, runs `MeshStore::PlanCreate` for the
// whole batch (so arena reserves happen before any CreateMesh), and then drains into ECS. Save
// reverses the flow, gathering per-(SourceMeshIndex, MeshKind) ECS state into one carrier per
// source mesh before emitting fastgltf primitives. PlanCreate's batch-then-commit contract is
// what blocks a single-pass collapse here — the bulk vertex/face data legitimately outlives the
// parse.
namespace gltf {
struct MeshData {
    std::optional<::MeshData> Triangles, Lines, Points;
    ::MeshVertexAttributes TriangleAttrs, LineAttrs, PointAttrs;
    ::MeshPrimitives TrianglePrimitives;
    std::optional<ArmatureDeformData> DeformData;
    std::optional<MorphTargetData> MorphData;
    std::string Name;
};

// `Node` and `Object` carry per-node state too entangled to dissolve. Load fans each Node into
// ~10 downstream passes (objects, parents, stubs, physics rb/joint, skin anchor, morph binding,
// animation resolve, joint-name reconciliation) interleaved with ECS construction. Save
// accumulates from independent ECS views and drains to `asset.nodes` only after the instance
// fan-in is known — itself gated on `node_objects` and `skin_remap`. Splitting fields hits
// option-1 (parallel cousins) or option-3 (tuples for MaterialRefs/TriggerData/JointData).
// `Object` is the EXT_mesh_gpu_instancing fan-out (one Node → N Objects) feeding the same
// passes, so it shares Node's fate.
struct Node {
    uint32_t NodeIndex;
    std::optional<uint32_t> ParentNodeIndex;
    std::vector<uint32_t> ChildrenNodeIndices;
    Transform LocalTransform, WorldTransform;
    bool InScene, IsJoint;
    std::optional<mat4> SourceMatrix{};
    std::optional<uint32_t> MeshIndex, SkinIndex, CameraIndex, LightIndex;
    std::string Name;

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
    Transform WorldTransform;
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
            const float outer = light.outerConeAngle ? *light.outerConeAngle : std::numbers::pi_v<float> / 4.f;
            const float inner = std::clamp(light.innerConeAngle ? *light.innerConeAngle : 0.f, 0.f, outer);
            pl.Type = PunctualLightType::Spot;
            pl.Range = light.range ? *light.range : 0.f;
            pl.InnerConeCos = std::cos(inner);
            pl.OuterConeCos = std::cos(outer);
            break;
        }
    }
    return pl;
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

std::expected<PopulateResult, std::string> LoadGltfFile(const std::filesystem::path &source_path, PopulateContext ctx) {
    const Timer timer{"LoadGltfFile"};

    ExtrasMap extras;
    auto parsed_asset = ParseAsset(source_path, &extras);
    if (!parsed_asset) return std::unexpected{parsed_asset.error()};

    auto &asset = *parsed_asset;
    if (asset.scenes.empty()) return std::unexpected{std::format("glTF '{}' has no scenes.", source_path.string())};

    const auto scene_index = asset.defaultScene.value_or(0);
    if (scene_index >= asset.scenes.size()) return std::unexpected{std::format("glTF '{}' has invalid default scene index.", source_path.string())};

    std::vector<Sampler> samplers;
    samplers.reserve(asset.samplers.size());
    for (const auto &sampler : asset.samplers) {
        samplers.emplace_back(Sampler{
            .MagFilter = ToFilter(sampler.magFilter),
            .MinFilter = ToFilter(sampler.minFilter),
            .WrapS = ToWrap(sampler.wrapS),
            .WrapT = ToWrap(sampler.wrapT),
            .Name = std::string(sampler.name),
        });
    }

    std::vector<Image> images;
    images.reserve(asset.images.size());
    for (uint32_t image_index = 0; image_index < asset.images.size(); ++image_index) {
        auto image_result = ReadImage(asset, image_index, source_path.parent_path());
        if (!image_result) return std::unexpected{std::move(image_result.error())};
        images.emplace_back(std::move(*image_result));
    }

    std::vector<Texture> textures;
    textures.reserve(asset.textures.size());
    for (const auto &texture : asset.textures) {
        textures.emplace_back(Texture{
            .SamplerIndex = ToIndex(texture.samplerIndex, asset.samplers.size()),
            .ImageIndex = ToIndex(texture.imageIndex, asset.images.size()),
            .WebpImageIndex = ToIndex(texture.webpImageIndex, asset.images.size()),
            .BasisuImageIndex = ToIndex(texture.basisuImageIndex, asset.images.size()),
            .DdsImageIndex = ToIndex(texture.ddsImageIndex, asset.images.size()),
            .Name = std::string(texture.name),
        });
    }

    // Plain `MaterialData` per source material — the name is read straight from `asset.materials`
    // when needed. The trailing entry is the synthetic "DefaultMaterial" fallback (see below).
    std::vector<MaterialData> source_materials;
    source_materials.reserve(asset.materials.size() + 1u);
    // Build the FromGpu-lossy delta per material in the same pass: extension-block presence,
    // KHR_texture_transform meta, source texture slots (pre-remap), and emissive_strength split.
    std::vector<MaterialSourceMeta> material_metas;
    material_metas.reserve(asset.materials.size() + 1u);
    const auto opt_slot = [](const auto &opt, auto field) -> uint32_t {
        return opt ? ((*opt).*field).Slot : InvalidSlot;
    };
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

        using M = MaterialSourceMeta;
        M meta{
            .EmissiveStrength = data.EmissiveStrength,
            .BaseSlotMeta = {data.BaseColorMeta, data.MetallicRoughnessMeta, data.NormalMeta, data.OcclusionMeta, data.EmissiveMeta},
            .NameWasEmpty = material.name.empty(),
        };
        meta.TextureSlots = {
            data.BaseColorTexture.Slot,
            data.MetallicRoughnessTexture.Slot,
            data.NormalTexture.Slot,
            data.OcclusionTexture.Slot,
            data.EmissiveTexture.Slot,
            opt_slot(data.Specular, &::Specular::Texture),
            opt_slot(data.Specular, &::Specular::ColorTexture),
            opt_slot(data.Sheen, &::Sheen::ColorTexture),
            opt_slot(data.Sheen, &::Sheen::RoughnessTexture),
            opt_slot(data.Transmission, &::Transmission::Texture),
            opt_slot(data.DiffuseTransmission, &::DiffuseTransmission::Texture),
            opt_slot(data.DiffuseTransmission, &::DiffuseTransmission::ColorTexture),
            opt_slot(data.Volume, &::Volume::ThicknessTexture),
            opt_slot(data.Clearcoat, &::Clearcoat::Texture),
            opt_slot(data.Clearcoat, &::Clearcoat::RoughnessTexture),
            opt_slot(data.Clearcoat, &::Clearcoat::NormalTexture),
            opt_slot(data.Anisotropy, &::Anisotropy::Texture),
            opt_slot(data.Iridescence, &::Iridescence::Texture),
            opt_slot(data.Iridescence, &::Iridescence::ThicknessTexture),
        };
        meta.ExtensionPresence = uint16_t(
            (data.Ior ? M::ExtIor : 0) | (data.Dispersion ? M::ExtDispersion : 0) |
            (data.EmissiveStrength ? M::ExtEmissiveStrength : 0) |
            (data.Sheen ? M::ExtSheen : 0) | (data.Specular ? M::ExtSpecular : 0) |
            (data.Transmission ? M::ExtTransmission : 0) | (data.DiffuseTransmission ? M::ExtDiffuseTransmission : 0) |
            (data.Volume ? M::ExtVolume : 0) | (data.Clearcoat ? M::ExtClearcoat : 0) |
            (data.Anisotropy ? M::ExtAnisotropy : 0) | (data.Iridescence ? M::ExtIridescence : 0)
        );
        material_metas.emplace_back(std::move(meta));
        source_materials.emplace_back(std::move(data));
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

    // Parse KHR_physics_rigid_bodies document-level resources. Collision filters are read straight
    // from `asset.collisionFilters` at the consumer site below — no intermediate staging vector.
    std::vector<PhysicsMaterial> physics_materials;
    std::vector<PhysicsJointDef> physics_joint_defs;
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
            physics_materials.emplace_back(PhysicsMaterial{
                .StaticFriction = float(src.staticFriction),
                .DynamicFriction = float(src.dynamicFriction),
                .Restitution = float(src.restitution),
                .FrictionCombine = ToCombineMode(src.frictionCombine),
                .RestitutionCombine = ToCombineMode(src.restitutionCombine),
            });
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
            physics_joint_defs.emplace_back(std::move(def));
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

    std::vector<Node> source_nodes;
    source_nodes.resize(asset.nodes.size());
    for (uint32_t node_index = 0; node_index < asset.nodes.size(); ++node_index) {
        const auto &source_node = asset.nodes[node_index];
        const auto source_mesh_index = ToIndex(source_node.meshIndex, asset.meshes.size());
        // Preserve the mesh ref verbatim - runtime gates rendering on InScene, not on MeshIndex,
        // so carrying a ref on out-of-source nodes is harmless and keeps the JSON round-trip honest.
        const auto mesh_index = source_mesh_index;
        std::vector<uint32_t> children_node_indices;
        children_node_indices.reserve(source_node.children.size());
        for (const auto child_idx : source_node.children) {
            if (const auto child = ToIndex(child_idx, asset.nodes.size())) children_node_indices.emplace_back(*child);
        }
        auto &node = source_nodes[node_index];
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
                // source_meshes is index-aligned with asset.meshes, so the glTF mesh index is also the source mesh index.
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
    for (uint32_t node_index = 0; node_index < source_nodes.size(); ++node_index) {
        if (const auto &node = source_nodes[node_index]; node.InScene) {
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

    std::vector<Object> source_objects;
    for (uint32_t node_index = 0; node_index < source_nodes.size(); ++node_index) {
        if (const auto &node = source_nodes[node_index]; is_object_emitted[node_index]) {
            const auto source_mesh_index = ToIndex(asset.nodes[node_index].meshIndex, asset.meshes.size());
            if (const auto &instance_transforms = node_instance_transforms[node_index];
                !instance_transforms.empty() && node.MeshIndex) {
                // EXT_mesh_gpu_instancing: emit one object per instance with baked world transform
                const auto base_name = MakeNodeName(asset, node.NodeIndex, source_mesh_index);
                const auto &source_weights = asset.nodes[node_index].weights;
                auto node_weights = source_weights.empty() ? std::optional<std::vector<float>>{} : std::optional{std::vector<float>(source_weights.begin(), source_weights.end())};
                for (uint32_t i = 0; i < instance_transforms.size(); ++i) {
                    // EXT_mesh_gpu_instancing: each instance is a root in the engine, so local == world.
                    auto instance_world = ToTransform(traversal.WorldTransforms[node_index] * ToMatrix(instance_transforms[i]));
                    source_objects.emplace_back(Object{
                        .ObjectType = Object::Type::Mesh,
                        .NodeIndex = node.NodeIndex,
                        .ParentNodeIndex = std::nullopt,
                        .LocalTransform = instance_world,
                        .WorldTransform = instance_world,
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
                source_objects.emplace_back(Object{
                    .ObjectType = object_type,
                    .NodeIndex = node.NodeIndex,
                    .ParentNodeIndex = nearest_object_ancestor[node.NodeIndex],
                    .LocalTransform = node.LocalTransform,
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

    // Snapshot source-form gltf data onto SceneEntity before the materials loop remaps `tex.Slot`
    // from gltf texture index to bindless sampler slot. AnimationOrder is patched in below after
    // animations are parsed (it depends on which source animations produced any valid channels).
    auto source_ibl = ConvertIBL(asset, scene_index);
    GltfSourceAssets source_assets{
        .Copyright = asset.assetInfo ? std::string(asset.assetInfo->copyright) : std::string{},
        .Generator = asset.assetInfo ? std::string(asset.assetInfo->generator) : std::string{},
        .MinVersion = asset.assetInfo ? std::string(asset.assetInfo->minVersion) : std::string{},
        .AssetExtras = asset.assetInfo ? std::string(asset.assetInfo->extras) : std::string{},
        .AssetExtensions = asset.assetInfo ? std::string(asset.assetInfo->extensions) : std::string{},
        .DefaultSceneName = std::string(asset.scenes[scene_index].name),
        .ExtrasByEntity = std::move(extras),
        .Textures = textures,
        .Images = images,
        .Samplers = samplers,
        .ImageBasedLight = source_ibl,
    };
    source_assets.DefaultSceneRoots.reserve(asset.scenes[scene_index].nodeIndices.size());
    for (const auto n : asset.scenes[scene_index].nodeIndices) {
        if (const auto idx = ToIndex(n, asset.nodes.size())) source_assets.DefaultSceneRoots.emplace_back(*idx);
    }
    source_assets.ExtensionsRequired.reserve(asset.extensionsRequired.size());
    for (const auto &e : asset.extensionsRequired) source_assets.ExtensionsRequired.emplace_back(e);
    source_assets.MaterialVariants.reserve(asset.materialVariants.size());
    for (const auto &v : asset.materialVariants) source_assets.MaterialVariants.emplace_back(v);
    source_assets.MaterialMetas = std::move(material_metas);
    R.emplace_or_replace<GltfSourceAssets>(SceneEntity, std::move(source_assets));

    std::vector<PendingTextureUpload> new_pending_textures;
    std::unordered_map<uint64_t, uint32_t> texture_slot_cache;
    // Cache on the resolved (image_index, sampler_index, color_space) rather than glTF texture index,
    // so that multiple glTF textures referencing the same image+sampler share a single TextureEntry and sampler slot.
    const auto texture_cache_key = [](uint32_t image_index, uint32_t sampler_index, TextureColorSpace color_space) {
        return (uint64_t(image_index) << 33u) | (uint64_t(sampler_index) << 1u) | (color_space == TextureColorSpace::Srgb ? 1u : 0u);
    };
    const auto resolve_texture_slot = [&](uint32_t texture_index, TextureColorSpace color_space) -> std::expected<uint32_t, std::string> {
        if (texture_index >= textures.size()) return InvalidSlot;

        const auto &src_texture = textures[texture_index];
        const auto image_index = resolve_image_index(src_texture);
        if (!image_index || *image_index >= images.size()) return InvalidSlot;

        const auto sampler_index = src_texture.SamplerIndex.value_or(InvalidSlot);
        const auto cache_key = texture_cache_key(*image_index, sampler_index, color_space);
        if (const auto it = texture_slot_cache.find(cache_key); it != texture_slot_cache.end()) return it->second;

        const auto *src_sampler = src_texture.SamplerIndex && *src_texture.SamplerIndex < samplers.size() ?
            &samplers[*src_texture.SamplerIndex] :
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
        auto gpu_material = gltf::ToGpu(src_material);
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
    // Track morph data per mesh for later component setup
    std::vector<std::optional<MorphTargetData>> mesh_morphs;
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
            // Detect PBR extension features before material index remapping.
            PbrFeatureMask mesh_pbr_mask{0};
            for (const auto gltf_mat_idx : scene_mesh.TrianglePrimitives.MaterialIndices) {
                if (gltf_mat_idx < source_materials.size()) {
                    const auto mat = gltf::ToGpu(source_materials[gltf_mat_idx]);
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
            // discards them. Stored as a sidecar so `BuildGltfScene` can re-emit verbatim.
            MeshSourceLayout layout{
                .VertexCounts = scene_mesh.TrianglePrimitives.VertexCounts,
                .AttributeFlags = scene_mesh.TrianglePrimitives.AttributeFlags,
                .HasSourceIndices = scene_mesh.TrianglePrimitives.HasSourceIndices,
                .VariantMappings = scene_mesh.TrianglePrimitives.VariantMappings,
                .Colors0ComponentCount = scene_mesh.TriangleAttrs.Colors0ComponentCount,
                .MorphTangentDeltas = scene_mesh.MorphData ? scene_mesh.MorphData->TangentDeltas : std::vector<vec3>{},
            };
            auto morph_data_copy = scene_mesh.MorphData; // Keep a copy for component setup
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
            if (!scene_mesh.Name.empty()) R.emplace<MeshName>(mesh_entity, MeshName{scene_mesh.Name});
            if (mesh_pbr_mask != 0) R.emplace<PbrMeshFeatures>(mesh_entity, mesh_pbr_mask);
            mesh_morphs.emplace_back(std::move(morph_data_copy));
        } else {
            mesh_morphs.emplace_back(std::nullopt);
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
            if (!scene_mesh.Name.empty()) R.emplace<MeshName>(e, MeshName{scene_mesh.Name});
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
            if (!cam.name.empty()) R.emplace<CameraName>(object_entity, std::string(cam.name));
        } else if (object.ObjectType == gltf::Object::Type::Light && object.LightIndex && *object.LightIndex < asset.lights.size()) {
            const auto &light = asset.lights[*object.LightIndex];
            object_entity = ::AddLight(R, ctx.Meshes, ctx.Buffers, SceneEntity, {.Name = object_name, .Transform = object.LocalTransform, .Select = MeshInstanceCreateInfo::SelectBehavior::None}, ConvertLight(light));
            R.emplace<SourceLightIndex>(object_entity, *object.LightIndex);
            if (!light.name.empty()) R.emplace<LightName>(object_entity, std::string(light.name));
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
        // empty; `source_nodes` preserves the raw value. Capture source-empty / collision-renamed.
        if (object.NodeIndex < source_nodes.size()) {
            const auto &raw_name = source_nodes[object.NodeIndex].Name;
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

    std::unordered_map<uint32_t, const gltf::Node *> scene_nodes_by_index;
    scene_nodes_by_index.reserve(source_nodes.size());
    for (const auto &node : source_nodes) scene_nodes_by_index.emplace(node.NodeIndex, &node);

    // Stubs for out-of-scene nodes (referenced only by non-default scenes) so build emits them
    // like the file round-trip does. They carry only what build needs and aren't in
    // `object_entities_by_node`, so runtime systems that walk the scene tree don't see them.
    for (const auto &node : source_nodes) {
        if (node.InScene) continue;
        const auto e = R.create();
        R.emplace<SourceNodeIndex>(e, node.NodeIndex);
        R.emplace<Transform>(e, node.LocalTransform);
        R.emplace<WorldTransform>(e, node.WorldTransform);
        if (node.MeshIndex && *node.MeshIndex < mesh_entities.size() && mesh_entities[*node.MeshIndex] != entt::null) {
            R.emplace<Instance>(e, mesh_entities[*node.MeshIndex]);
        }
        if (node.Name.empty()) R.emplace<SourceEmptyName>(e);
        else R.emplace<Name>(e, node.Name);
    }

    // KHR_physics_rigid_bodies: promote loader's index-keyed resource arrays to registry entities,
    // each tagged with its source index for round-trip ordering.
    {
        const auto promote = [&]<typename TComp, typename TIndex>(const auto &src) {
            std::vector<entt::entity> entities;
            entities.reserve(src.size());
            for (uint32_t i = 0; i < src.size(); ++i) {
                const auto e = R.create();
                R.emplace<TComp>(e, src[i]);
                R.emplace<TIndex>(e, i);
                entities.emplace_back(e);
            }
            return entities;
        };
        const auto material_entities = promote.operator()<PhysicsMaterial, SourcePhysicsMaterialIndex>(physics_materials);
        const auto jointdef_entities = promote.operator()<::PhysicsJointDef, SourcePhysicsJointDefIndex>(physics_joint_defs);

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
            CollisionFilter filter{.Systems = resolve_systems(src.collisionSystems)};
            // KHR schema forbids both collideWith and notCollideWith; prefer allowlist if both appear.
            if (!src.collideWithSystems.empty()) {
                filter.Mode = CollideMode::Allowlist;
                filter.CollideSystems = resolve_systems(src.collideWithSystems);
            } else if (!src.notCollideWithSystems.empty()) {
                filter.Mode = CollideMode::Blocklist;
                filter.CollideSystems = resolve_systems(src.notCollideWithSystems);
            }
            const auto e = R.create();
            R.emplace<CollisionFilter>(e, std::move(filter));
            R.emplace<SourceCollisionFilterIndex>(e, i);
            filter_entities.emplace_back(e);
        }

        auto resolve_mat = [&](std::optional<uint32_t> idx) {
            return idx && *idx < material_entities.size() ? material_entities[*idx] : null_entity;
        };
        auto resolve_filter = [&](std::optional<uint32_t> idx) {
            return idx && *idx < filter_entities.size() ? filter_entities[*idx] : null_entity;
        };

        for (const auto &node : source_nodes) {
            auto it = object_entities_by_node.find(node.NodeIndex);
            if (it == object_entities_by_node.end()) continue;
            const auto entity = it->second;

            if (node.Collider) {
                auto collider = *node.Collider;
                if (IsMeshBackedShape(collider.Shape)) {
                    if (node.ColliderGeometryMeshIndex && *node.ColliderGeometryMeshIndex < mesh_entities.size()) {
                        collider.MeshEntity = mesh_entities[*node.ColliderGeometryMeshIndex];
                    } else if (R.all_of<Instance>(entity)) {
                        collider.MeshEntity = R.get<const Instance>(entity).Entity;
                    }
                }
                R.emplace<ColliderShape>(entity, std::move(collider));
                // Imported collider state is authoritative — engine must not auto-derive over it.
                R.emplace<ColliderPolicy>(entity, ColliderPolicy{.AutoFitDims = false, .LockedKind = true});
                if (node.Material) {
                    R.replace<ColliderMaterial>(entity, ColliderMaterial{
                                                            .PhysicsMaterialEntity = resolve_mat(node.Material->PhysicsMaterialIndex),
                                                            .CollisionFilterEntity = resolve_filter(node.Material->CollisionFilterIndex),
                                                        });
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
                        ColliderShape shape{.Shape = *td.Shape};
                        if (td.GeometryMeshIndex && *td.GeometryMeshIndex < mesh_entities.size()) {
                            shape.MeshEntity = mesh_entities[*td.GeometryMeshIndex];
                        }
                        R.emplace<ColliderShape>(entity, std::move(shape));
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
                const auto def_entity = jd.JointDefIndex < jointdef_entities.size() ? jointdef_entities[jd.JointDefIndex] : null_entity;
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

        const auto anchor_it = scene_nodes_by_index.find(*anchor_node_index);
        if (anchor_it == scene_nodes_by_index.end() || !anchor_it->second->InScene) {
            return std::unexpected{std::format("glTF import failed for '{}': skin {} anchor node {} is not in the imported scene.", source_path.string(), skin_index, *anchor_node_index)};
        }

        const std::string skin_name(skin.name);
        const auto armature_entity = R.create();
        R.emplace<ObjectKind>(armature_entity, ObjectType::Armature);
        R.emplace<ArmatureObject>(armature_entity, armature_data_entity);
        const auto &t = anchor_it->second->WorldTransform;
        R.emplace<Transform>(armature_entity, t);
        R.emplace<WorldTransform>(armature_entity, t);
        R.emplace<Name>(armature_entity, ::CreateName(R, SceneEntity, skin_name.empty() ? std::format("{}_Armature{}", name_prefix, skin_index) : skin_name));
        if (skin_name.empty()) R.emplace<SourceEmptyName>(armature_entity);
        else R.emplace<SkinName>(armature_entity, SkinName{skin_name});

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

        { // Create pose state — GPU deform buffer allocation deferred to ProcessComponentEvents.
            ArmaturePoseState pose_state;
            pose_state.BonePoseDelta.resize(armature.Bones.size(), Transform{});
            pose_state.BoneUserOffset.resize(armature.Bones.size(), Transform{});
            pose_state.BonePoseWorld.resize(armature.Bones.size(), I4);
            R.emplace<ArmaturePoseState>(armature_data_entity, std::move(pose_state));
        }
        ::CreateBoneInstances(R, ctx.Meshes, SceneEntity, armature_entity, armature_data_entity);
        // Mark each bone entity with its source joint NodeIndex (for SaveScene round-trip).
        const auto &bone_entities_for_source = R.get<const ArmatureObject>(armature_entity).BoneEntities;
        for (uint32_t i = 0; i < armature.Bones.size(); ++i) {
            const auto joint_node_index = armature.Bones[i].JointNodeIndex;
            if (!joint_node_index) continue;
            R.emplace<SourceNodeIndex>(bone_entities_for_source[i], *joint_node_index);
            if (*joint_node_index < source_nodes.size() && source_nodes[*joint_node_index].Name.empty()) {
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
                    const auto nit = scene_nodes_by_index.find(*cur);
                    if (nit == scene_nodes_by_index.end()) break;
                    cur = nit->second->ParentNodeIndex;
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
        const auto it = scene_nodes_by_index.find(sni.Value);
        if (it == scene_nodes_by_index.end()) continue;
        const auto &source_node = *it->second;
        if (source_node.ParentNodeIndex) {
            R.emplace<SourceParentNodeIndex>(entity, *source_node.ParentNodeIndex);
            if (const auto parent_it = scene_nodes_by_index.find(*source_node.ParentNodeIndex); parent_it != scene_nodes_by_index.end()) {
                const auto &siblings = parent_it->second->ChildrenNodeIndices;
                if (const auto pos = std::ranges::find(siblings, sni.Value); pos != siblings.end()) {
                    R.emplace<SourceSiblingIndex>(entity, uint32_t(pos - siblings.begin()));
                }
            }
        }
        if (source_node.SourceMatrix) R.emplace<SourceMatrixTransform>(entity, *source_node.SourceMatrix);
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
        if (*object.MeshIndex >= mesh_morphs.size() || !mesh_morphs[*object.MeshIndex]) continue;
        const auto obj_it = object_entities_by_node.find(object.NodeIndex);
        if (obj_it == object_entities_by_node.end()) continue;
        const auto instance_entity = obj_it->second;
        if (!R.all_of<Instance>(instance_entity)) continue;

        const auto &morph = *mesh_morphs[*object.MeshIndex];
        if (morph.TargetCount == 0) continue;

        MorphWeightState state;
        if (object.NodeWeights) {
            state.Weights.resize(morph.TargetCount, 0.f);
            std::copy_n(object.NodeWeights->begin(), std::min(uint32_t(object.NodeWeights->size()), morph.TargetCount), state.Weights.begin());
        } else {
            state.Weights = morph.DefaultWeights;
        }
        R.emplace<MorphWeightState>(instance_entity, std::move(state));
        morph_instance_by_node[object.NodeIndex] = instance_entity;
    }

    // Resolve object/node transform animations (empties, meshes, cameras, lights).
    // Channels targeting skin joints are handled by ArmatureAnimation and skipped here.
    std::unordered_map<entt::entity, Transform> node_anim_bindings;
    node_anim_bindings.reserve(object_entities_by_node.size());
    for (const auto &[node_index, object_entity] : object_entities_by_node) {
        if (!R.valid(object_entity)) continue;
        const auto node_it = scene_nodes_by_index.find(node_index);
        if (node_it == scene_nodes_by_index.end()) continue;
        node_anim_bindings.emplace(object_entity, node_it->second->LocalTransform);
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
    R.patch<GltfSourceAssets>(SceneEntity, [&](auto &a) { a.AnimationOrder = std::move(animation_order); });

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

    return gltf::PopulateResult{
        .FirstMesh = first_mesh_entity,
        .Active = active_entity,
        .FirstCameraObject = first_camera_object_entity,
        .ImportedAnimation = imported_animation,
    };
}

std::expected<void, std::string> SaveGltfFile(const entt::registry &r, entt::entity scene_entity, const SceneBuffers &buffers, const MeshStore &meshes, const std::filesystem::path &path) {
    const Timer timer{"SaveGltfFile"};

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

    // Source-form scene metadata + texture/image/sampler arrays come from the GltfSourceAssets
    // sidecar — encoded image bytes, sampler-config collapse, and asset.* metadata aren't
    // recoverable from registry/GPU state. Cameras and lights emit below from per-entity
    // components (see CameraName/LightName). Materials reconstruct via FromGpu(GPU buffer) +
    // patch from the per-material delta in `MaterialMetas`.
    const auto *src_assets = r.try_get<const GltfSourceAssets>(scene_entity);
    // Source-form scene-level metadata (Copyright/Generator/MinVersion/Asset.*, DefaultScene,
    // ExtensionsRequired/MaterialVariants/ExtrasByEntity, Textures/Images/Samplers/IBL) is read
    // directly from `src_assets` at each emit site below (no intermediate aggregate).
    static const GltfSourceAssets EmptySourceAssets{};
    const auto &sa = src_assets ? *src_assets : EmptySourceAssets;
    // Materials: skip the engine "Default" at registry index 0; loaded gltf materials live at
    // [1, count). Reconstruct each via `FromGpu` then patch with `MaterialSourceMeta` to restore
    // KHR_materials_emissive_strength split, KHR_texture_transform meta, source texture indices,
    // and the optionality of extension blocks (FromGpu's value-vs-default gate is lossy for them).
    const auto &names = r.get<const MaterialStore>(scene_entity).Names;
    const auto material_count = buffers.Materials.Count();
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

    // Source meshes → gltf::MeshData. Triangles populate the .Triangles slot below, Lines and
    // Points entities (which share a SourceMeshIndex with their sibling kinds) populate the
    // .Lines / .Points slots in the second pass.
    std::vector<gltf::MeshData> save_meshes(source_mesh_count);
    for (const auto [entity, smi, layout] : r.view<const SourceMeshIndex, const MeshSourceLayout>().each()) {
        const auto *mesh_ptr = r.try_get<const Mesh>(entity);
        if (!mesh_ptr) continue;
        const auto &mesh = *mesh_ptr;
        const auto store_id = mesh.GetStoreId();
        const auto vertices = meshes.GetVertices(store_id);
        const auto vertex_count = uint32_t(vertices.size());

        ::MeshData triangles;
        triangles.Positions.reserve(vertex_count);
        for (const auto &v : vertices) triangles.Positions.emplace_back(v.Position);
        triangles.Faces.reserve(mesh.FaceCount());
        for (const auto fh : mesh.faces()) {
            std::vector<uint32_t> face_indices;
            for (const auto vh : mesh.fv_range(fh)) face_indices.emplace_back(*vh);
            triangles.Faces.emplace_back(std::move(face_indices));
        }

        // Populate a TriangleAttrs slot iff any source primitive had that bit set. Per-primitive
        // emission re-checks via `MeshPrimitives::AttributeFlags` at save time.
        uint32_t any_flags = 0;
        for (const auto f : layout.AttributeFlags) any_flags |= f;
        ::MeshVertexAttributes attrs;
        attrs.Colors0ComponentCount = layout.Colors0ComponentCount;
        const auto fill = [&]<typename V>(uint32_t bit, std::optional<std::vector<V>> &dest, V Vertex::*field) {
            if (!(any_flags & bit)) return;
            dest.emplace();
            dest->reserve(vertex_count);
            for (const auto &v : vertices) dest->emplace_back(v.*field);
        };
        fill(MeshAttributeBit_Normal, attrs.Normals, &Vertex::Normal);
        fill(MeshAttributeBit_Tangent, attrs.Tangents, &Vertex::Tangent);
        fill(MeshAttributeBit_Color0, attrs.Colors0, &Vertex::Color);
        fill(MeshAttributeBit_TexCoord0, attrs.TexCoords0, &Vertex::TexCoord0);
        fill(MeshAttributeBit_TexCoord1, attrs.TexCoords1, &Vertex::TexCoord1);
        fill(MeshAttributeBit_TexCoord2, attrs.TexCoords2, &Vertex::TexCoord2);
        fill(MeshAttributeBit_TexCoord3, attrs.TexCoords3, &Vertex::TexCoord3);

        const auto face_primitives = meshes.GetFacePrimitiveIndices(store_id);
        const auto primitive_materials = meshes.GetPrimitiveMaterialIndices(store_id);
        // Reverse populate's +1 material-index shift; `~0u` (out-of-range) = don't emit.
        ::MeshPrimitives prims{
            .FacePrimitiveIndices = {face_primitives.begin(), face_primitives.end()},
            .VertexCounts = layout.VertexCounts,
            .AttributeFlags = layout.AttributeFlags,
            .HasSourceIndices = layout.HasSourceIndices,
            .VariantMappings = layout.VariantMappings,
        };
        prims.MaterialIndices.reserve(primitive_materials.size());
        for (const auto reg_idx : primitive_materials) prims.MaterialIndices.emplace_back(reg_idx >= 1 ? reg_idx - 1 : ~0u);

        std::optional<ArmatureDeformData> deform_data;
        if (const auto bd_range = meshes.GetBoneDeformRange(store_id); bd_range.Count > 0) {
            const auto bd_span = meshes.BoneDeformBuffer.Get(bd_range);
            ArmatureDeformData dd;
            dd.Joints.reserve(bd_span.size());
            dd.Weights.reserve(bd_span.size());
            for (const auto &bdv : bd_span) {
                dd.Joints.emplace_back(bdv.Joints);
                dd.Weights.emplace_back(bdv.Weights);
            }
            deform_data = std::move(dd);
        }

        std::optional<MorphTargetData> morph_data;
        if (const auto target_count = meshes.GetMorphTargetCount(store_id); target_count > 0 && vertex_count > 0) {
            const auto mt_span = meshes.MorphTargetBuffer.Get(meshes.GetMorphTargetRange(store_id));
            MorphTargetData md{.TargetCount = target_count};
            md.PositionDeltas.reserve(mt_span.size());
            for (const auto &m : mt_span) md.PositionDeltas.emplace_back(m.PositionDelta);
            // CreateMesh writes 0 when source lacked normal deltas, so any non-zero ⇒ source had them.
            if (std::ranges::any_of(mt_span, [](const auto &m) { return m.NormalDelta != vec3{0}; })) {
                md.NormalDeltas.reserve(mt_span.size());
                for (const auto &m : mt_span) md.NormalDeltas.emplace_back(m.NormalDelta);
            }
            md.TangentDeltas = layout.MorphTangentDeltas;
            const auto default_weights = meshes.GetDefaultMorphWeights(store_id);
            md.DefaultWeights.assign(default_weights.begin(), default_weights.end());
            morph_data = std::move(md);
        }

        const auto *mn = r.try_get<const MeshName>(entity);
        save_meshes[smi.Value] = gltf::MeshData{
            .Triangles = std::move(triangles),
            .TriangleAttrs = std::move(attrs),
            .TrianglePrimitives = std::move(prims),
            .DeformData = std::move(deform_data),
            .MorphData = std::move(morph_data),
            .Name = mn ? mn->Value : std::string{},
        };
    }

    // Lines / Points entities → fill `.Lines` / `.Points` on the corresponding save_meshes slot.
    // Lines/Points entities → fill the matching `.Lines`/`.Points` slot. Attrs are recovered by
    // detecting non-default values: NORMAL non-zero, COLOR_0 not (1,1,1,1).
    for (const auto [entity, smi, kind] : r.view<const SourceMeshIndex, const SourceMeshKind>().each()) {
        if (kind.Value == MeshKind::Triangles) continue;
        const auto *mesh_ptr = r.try_get<const Mesh>(entity);
        if (!mesh_ptr) continue;
        const auto &mesh = *mesh_ptr;
        const auto vertices = meshes.GetVertices(mesh.GetStoreId());
        ::MeshData md;
        md.Positions.reserve(vertices.size());
        for (const auto &v : vertices) md.Positions.emplace_back(v.Position);
        if (kind.Value == MeshKind::Lines) {
            md.Edges.reserve(mesh.EdgeCount());
            for (const auto eh : mesh.edges()) {
                const auto h0 = mesh.GetHalfedge(eh, 0);
                md.Edges.emplace_back(std::array<uint32_t, 2>{*mesh.GetFromVertex(h0), *mesh.GetToVertex(h0)});
            }
        }
        ::MeshVertexAttributes attrs;
        const auto fill_if_any = [&]<typename V>(std::optional<std::vector<V>> &dest, V Vertex::*field, V sentinel) {
            if (!std::ranges::any_of(vertices, [&](const auto &v) { return v.*field != sentinel; })) return;
            dest.emplace();
            dest->reserve(vertices.size());
            for (const auto &v : vertices) dest->emplace_back(v.*field);
        };
        fill_if_any(attrs.Normals, &Vertex::Normal, vec3{0});
        fill_if_any(attrs.Colors0, &Vertex::Color, vec4{1});
        if (attrs.Colors0) attrs.Colors0ComponentCount = 4; // CPU stores vec4 regardless of source
        auto &dst = save_meshes[smi.Value];
        if (kind.Value == MeshKind::Lines) {
            dst.Lines = std::move(md);
            dst.LineAttrs = std::move(attrs);
        } else {
            dst.Points = std::move(md);
            dst.PointAttrs = std::move(attrs);
        }
        if (dst.Name.empty()) {
            if (const auto *mn = r.try_get<const MeshName>(entity)) dst.Name = mn->Value;
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

    std::vector<gltf::Node> save_nodes(total_node_count);
    for (const auto [entity, node_index] : entity_to_node_index) {
        auto &node = save_nodes[node_index];
        node.NodeIndex = node_index;
        node.LocalTransform = r.get<const Transform>(entity);
        node.WorldTransform = r.get<const WorldTransform>(entity);
        node.InScene = true;
        node.IsJoint = r.all_of<BoneIndex>(entity);
        if (const auto *smt = r.try_get<const SourceMatrixTransform>(entity)) node.SourceMatrix = smt->Value;
        if (!r.all_of<SourceEmptyName>(entity)) {
            if (const auto *son = r.try_get<const SourceObjectName>(entity)) node.Name = son->Value;
            else if (const auto *name = r.try_get<const Name>(entity)) node.Name = name->Value;
        }
        if (const auto *spi = r.try_get<const SourceParentNodeIndex>(entity)) node.ParentNodeIndex = spi->Value;
        if (const auto it = children_by_parent.find(node_index); it != children_by_parent.end()) {
            node.ChildrenNodeIndices.reserve(it->second.size());
            for (const auto &[_, child_idx] : it->second) node.ChildrenNodeIndices.push_back(child_idx);
        }
    }

    // Fill mesh/camera/light/skin refs on both nodes and objects from the entity-index maps.
    // `ArmatureModifier::ArmatureEntity` actually holds the Armature *data* entity (legacy naming).
    const auto fill_refs = [&](entt::entity entity, auto &dst) {
        if (const auto *inst = r.try_get<const Instance>(entity)) {
            if (const auto it = mesh_entity_to_index.find(inst->Entity); it != mesh_entity_to_index.end()) dst.MeshIndex = it->second;
        }
        if (const auto it = camera_entity_to_index.find(entity); it != camera_entity_to_index.end()) dst.CameraIndex = it->second;
        if (const auto it = light_entity_to_index.find(entity); it != light_entity_to_index.end()) dst.LightIndex = it->second;
        if (const auto *am = r.try_get<const ArmatureModifier>(entity)) {
            if (const auto it = armature_data_to_skin_index.find(am->ArmatureEntity); it != armature_data_to_skin_index.end()) dst.SkinIndex = it->second;
        }
    };
    for (const auto [entity, node_index] : entity_to_node_index) fill_refs(entity, save_nodes[node_index]);

    const auto to_object_type = [](ObjectType k) {
        switch (k) {
            case ObjectType::Mesh: return gltf::Object::Type::Mesh;
            case ObjectType::Camera: return gltf::Object::Type::Camera;
            case ObjectType::Light: return gltf::Object::Type::Light;
            default: return gltf::Object::Type::Empty;
        }
    };
    std::vector<gltf::Object> save_objects;
    auto object_view = r.view<const Transform, const ObjectKind>();
    for (const auto entity : object_view) {
        const auto kind = object_view.get<const ObjectKind>(entity).Value;
        if (kind == ObjectType::Armature) continue; // → gltf::Skin, handled separately.
        const auto it = entity_to_node_index.find(entity);
        if (it == entity_to_node_index.end()) continue;
        const auto *spi = r.try_get<const SourceParentNodeIndex>(entity);
        gltf::Object obj{
            .ObjectType = to_object_type(kind),
            .NodeIndex = it->second,
            .ParentNodeIndex = spi ? std::optional{spi->Value} : std::nullopt,
            .WorldTransform = r.get<const WorldTransform>(entity),
        };
        if (const auto *name = r.try_get<const Name>(entity)) obj.Name = name->Value;
        fill_refs(entity, obj);
        save_objects.emplace_back(std::move(obj));
    }

    // Asset is declared early so collision filters can be emitted directly into
    // `asset.collisionFilters` (their pmr allocator binds to the asset).
    fastgltf::Asset asset;

    // Physics document-level resources, source-aligned via the per-resource index sidecars.
    std::unordered_map<entt::entity, uint32_t> physics_material_to_index, physics_jointdef_to_index, collision_filter_to_index;
    std::vector<PhysicsMaterial> save_physics_materials;
    std::vector<::PhysicsJointDef> save_physics_joint_defs;
    {
        auto mat_view = r.view<const PhysicsMaterial>();
        for (const auto &[_, e] : ordered_by_source.operator()<SourcePhysicsMaterialIndex>(mat_view)) {
            physics_material_to_index[e] = uint32_t(save_physics_materials.size());
            save_physics_materials.emplace_back(mat_view.get<const PhysicsMaterial>(e));
        }
        auto jd_view = r.view<const ::PhysicsJointDef>();
        for (const auto &[_, e] : ordered_by_source.operator()<SourcePhysicsJointDefIndex>(jd_view)) {
            physics_jointdef_to_index[e] = uint32_t(save_physics_joint_defs.size());
            save_physics_joint_defs.emplace_back(jd_view.get<const ::PhysicsJointDef>(e));
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
            collision_filter_to_index[e] = uint32_t(asset.collisionFilters.size());
            fastgltf::CollisionFilter out{.collisionSystems = resolve_system_names(f.Systems)};
            if (f.Mode == CollideMode::Allowlist) out.collideWithSystems = resolve_system_names(f.CollideSystems);
            else if (f.Mode == CollideMode::Blocklist) out.notCollideWithSystems = resolve_system_names(f.CollideSystems);
            asset.collisionFilters.emplace_back(std::move(out));
        }
    }

    // Per-node physics: walk all nodes, fill Motion/Velocity/Collider/Material/Trigger/Joint.
    for (const auto [entity, node_index] : entity_to_node_index) {
        if (node_index >= save_nodes.size()) continue;
        auto &node = save_nodes[node_index];
        if (const auto *m = r.try_get<const PhysicsMotion>(entity)) node.Motion = *m;
        if (const auto *v = r.try_get<const PhysicsVelocity>(entity)) node.Velocity = *v;
        if (const auto *cs = r.try_get<const ColliderShape>(entity)) {
            if (!r.all_of<TriggerTag>(entity)) {
                node.Collider = *cs;
                if (const auto *cm = r.try_get<const ColliderMaterial>(entity)) {
                    gltf::Node::MaterialRefs refs;
                    if (const auto mit = physics_material_to_index.find(cm->PhysicsMaterialEntity); mit != physics_material_to_index.end()) refs.PhysicsMaterialIndex = mit->second;
                    if (const auto fit = collision_filter_to_index.find(cm->CollisionFilterEntity); fit != collision_filter_to_index.end()) refs.CollisionFilterIndex = fit->second;
                    if (refs.PhysicsMaterialIndex || refs.CollisionFilterIndex) node.Material = refs;
                }
                if (cs->MeshEntity != null_entity) {
                    if (const auto mit = mesh_entity_to_index.find(cs->MeshEntity); mit != mesh_entity_to_index.end()) node.ColliderGeometryMeshIndex = mit->second;
                }
            } else {
                // Geometry trigger: shape on the same entity, distinguished by TriggerTag.
                gltf::Node::TriggerData td{.Shape = cs->Shape};
                if (cs->MeshEntity != null_entity) {
                    if (const auto mit = mesh_entity_to_index.find(cs->MeshEntity); mit != mesh_entity_to_index.end()) td.GeometryMeshIndex = mit->second;
                }
                if (const auto *cm = r.try_get<const ColliderMaterial>(entity)) {
                    if (const auto fit = collision_filter_to_index.find(cm->CollisionFilterEntity); fit != collision_filter_to_index.end()) td.CollisionFilterIndex = fit->second;
                }
                node.Trigger = std::move(td);
            }
        }
        if (const auto *tn = r.try_get<const TriggerNodes>(entity)) {
            gltf::Node::TriggerData td;
            td.NodeIndices.reserve(tn->Nodes.size());
            for (const auto ne : tn->Nodes) {
                if (const auto nit = entity_to_node_index.find(ne); nit != entity_to_node_index.end()) td.NodeIndices.emplace_back(nit->second);
            }
            if (const auto fit = collision_filter_to_index.find(tn->CollisionFilterEntity); fit != collision_filter_to_index.end()) td.CollisionFilterIndex = fit->second;
            node.Trigger = std::move(td);
        }
        if (const auto *pj = r.try_get<const PhysicsJoint>(entity)) {
            gltf::Node::JointData jd;
            if (const auto cit = entity_to_node_index.find(pj->ConnectedNode); cit != entity_to_node_index.end()) jd.ConnectedNodeIndex = cit->second;
            if (const auto dit = physics_jointdef_to_index.find(pj->JointDefEntity); dit != physics_jointdef_to_index.end()) jd.JointDefIndex = dit->second;
            jd.EnableCollision = pj->EnableCollision;
            node.Joint = jd;
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

    // Animations: merge per-entity engine clips back into source-side clips by (Name, Duration).
    // Engine splits each source clip into N entity clips (one per affected entity); reverse here.
    // Pre-seed `asset.animations` in source-name order from `GltfSourceAssets::AnimationOrder` so
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
        anim.samplers.emplace_back(fastgltf::AnimationSampler{
            .inputAccessor = t_acc,
            .outputAccessor = v_acc,
            .interpolation = FromInterp(interp),
        });
        anim.channels.emplace_back(fastgltf::AnimationChannel{
            .samplerIndex = anim.samplers.size() - 1,
            .nodeIndex = target_node_index,
            .path = FromPath(target),
        });
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

    // Images: re-emit in the source's form - data URI, external URI, or embedded bufferView.
    std::vector<uint32_t> image_bv_index(sa.Images.size(), UINT32_MAX);
    asset.images.reserve(sa.Images.size());
    for (uint32_t i = 0; i < sa.Images.size(); ++i) {
        const auto &img = sa.Images[i];
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
    // "DefaultMaterial" the loader appends — emit i in [1, material_count - 1).
    const uint32_t save_material_count = material_count > 1 ? material_count - 2u : 0u;
    asset.materials.reserve(save_material_count);
    using M = MaterialSourceMeta;
    // Restore an extension's optional<>: bit set + FromGpu produced nullopt (all-defaults block) → default-construct.
    // Bit clear → drop FromGpu's reconstruction (source had no extension).
    const auto sync_ext = [&]<typename T>(std::optional<T> &slot, uint16_t bits, uint16_t bit) {
        if (bits & bit) {
            if (!slot) slot = T{};
        } else slot.reset();
    };
    for (uint32_t i = 1; i <= save_material_count; ++i) {
        const auto source_idx = i - 1;
        auto data = gltf::FromGpu(buffers.Materials.Get(i));
        std::string name = i < names.size() ? names[i] : std::string{};
        if (source_idx < material_metas.size()) {
            const auto &meta = material_metas[source_idx];
            const auto bits = meta.ExtensionPresence;
            // Base texture slots + KHR_texture_transform meta.
            const std::array<TextureInfo *, 5> base_tex{&data.BaseColorTexture, &data.MetallicRoughnessTexture, &data.NormalTexture, &data.OcclusionTexture, &data.EmissiveTexture};
            const std::array<gltf::TextureTransformMeta *, 5> base_meta{&data.BaseColorMeta, &data.MetallicRoughnessMeta, &data.NormalMeta, &data.OcclusionMeta, &data.EmissiveMeta};
            for (uint8_t k = 0; k < 5; ++k) {
                base_tex[k]->Slot = meta.TextureSlots[k];
                *base_meta[k] = meta.BaseSlotMeta[k];
            }
            // Extension presence — Ior/Dispersion are scalars; the rest are sub-structs.
            if (bits & M::ExtIor) {
                if (!data.Ior) data.Ior = 1.5f;
            } else data.Ior.reset();
            if (bits & M::ExtDispersion) {
                if (!data.Dispersion) data.Dispersion = 0.f;
            } else data.Dispersion.reset();
            sync_ext(data.Sheen, bits, M::ExtSheen);
            sync_ext(data.Specular, bits, M::ExtSpecular);
            sync_ext(data.Transmission, bits, M::ExtTransmission);
            sync_ext(data.DiffuseTransmission, bits, M::ExtDiffuseTransmission);
            sync_ext(data.Volume, bits, M::ExtVolume);
            sync_ext(data.Clearcoat, bits, M::ExtClearcoat);
            sync_ext(data.Anisotropy, bits, M::ExtAnisotropy);
            sync_ext(data.Iridescence, bits, M::ExtIridescence);
            // Nested extension texture slots — assigns are no-ops when the optional is empty.
            const auto put = [&](auto &opt, auto field, MaterialTextureSlot s) {
                if (opt) ((*opt).*field).Slot = meta.TextureSlots[s];
            };
            put(data.Specular, &::Specular::Texture, MTS_Specular);
            put(data.Specular, &::Specular::ColorTexture, MTS_SpecularColor);
            put(data.Sheen, &::Sheen::ColorTexture, MTS_SheenColor);
            put(data.Sheen, &::Sheen::RoughnessTexture, MTS_SheenRoughness);
            put(data.Transmission, &::Transmission::Texture, MTS_Transmission);
            put(data.DiffuseTransmission, &::DiffuseTransmission::Texture, MTS_DiffuseTransmission);
            put(data.DiffuseTransmission, &::DiffuseTransmission::ColorTexture, MTS_DiffuseTransmissionColor);
            put(data.Volume, &::Volume::ThicknessTexture, MTS_VolumeThickness);
            put(data.Clearcoat, &::Clearcoat::Texture, MTS_Clearcoat);
            put(data.Clearcoat, &::Clearcoat::RoughnessTexture, MTS_ClearcoatRoughness);
            put(data.Clearcoat, &::Clearcoat::NormalTexture, MTS_ClearcoatNormal);
            put(data.Anisotropy, &::Anisotropy::Texture, MTS_Anisotropy);
            put(data.Iridescence, &::Iridescence::Texture, MTS_Iridescence);
            put(data.Iridescence, &::Iridescence::ThicknessTexture, MTS_IridescenceThickness);
            // ToGpu folded `EmissiveFactor *= strength`; un-fold for round-trip.
            if (meta.EmissiveStrength) {
                const float s = *meta.EmissiveStrength;
                if (s != 0.f) data.EmissiveFactor /= s;
                data.EmissiveStrength = s;
            }
            if (meta.NameWasEmpty) name.clear();
        }
        const auto &m = data;
        fastgltf::Material out;
        out.name = ToFgStr(name);

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

    asset.meshes.reserve(save_meshes.size());
    for (uint32_t mi = 0; mi < save_meshes.size(); ++mi) {
        const auto &mesh = save_meshes[mi];
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
                    if (mat < save_material_count) material_index = mat;
                }

                std::vector<fastgltf::Optional<std::size_t>> mappings;
                if (prim_idx < tprims.VariantMappings.size()) {
                    for (const auto &m : tprims.VariantMappings[prim_idx]) {
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

        skin_remap[source_skin_index] = uint32_t(asset.skins.size());
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

    // Nodes referenced by multiple Objects become EXT_mesh_gpu_instancing.
    std::vector<std::vector<uint32_t>> node_objects(save_nodes.size());
    for (uint32_t oi = 0; oi < save_objects.size(); ++oi) {
        const auto ni = save_objects[oi].NodeIndex;
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
    asset.nodes.reserve(save_nodes.size());
    bool uses_gpu_instancing = false;
    bool uses_physics_rigid_bodies = false;
    for (uint32_t ni = 0; ni < save_nodes.size(); ++ni) {
        const auto &n = save_nodes[ni];

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
                const auto &obj = save_objects[obj_indices[i]];
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
    if (!sa.DefaultSceneRoots.empty()) {
        scene_roots.reserve(sa.DefaultSceneRoots.size());
        for (const auto n : sa.DefaultSceneRoots) scene_roots.emplace_back(n);
    } else {
        for (uint32_t ni = 0; ni < save_nodes.size(); ++ni) {
            const auto &n = save_nodes[ni];
            if (!n.InScene) continue;
            if (n.ParentNodeIndex && *n.ParentNodeIndex < save_nodes.size() && save_nodes[*n.ParentNodeIndex].InScene) continue;
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

    // Physics doc-level arrays
    asset.physicsMaterials.reserve(save_physics_materials.size());
    for (const auto &pm : save_physics_materials) {
        asset.physicsMaterials.emplace_back(fastgltf::PhysicsMaterial{
            .staticFriction = pm.StaticFriction,
            .dynamicFriction = pm.DynamicFriction,
            .restitution = pm.Restitution,
            .frictionCombine = FromCombine(pm.FrictionCombine),
            .restitutionCombine = FromCombine(pm.RestitutionCombine),
        });
    }
    asset.physicsJoints.reserve(save_physics_joint_defs.size());
    for (const auto &jd : save_physics_joint_defs) {
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
