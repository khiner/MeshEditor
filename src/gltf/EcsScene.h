// Bridge between glTF files and the EnTT registry + canonical UMA buffers.
//
// `LoadGltfFile` parses a glTF/.glb, populates registry components, and writes to canonical UMA
// buffers (Materials, MeshStore arenas, TextureStore). Texture uploads and IBL prefilter currently
// run synchronously here; deferring them to `ProcessComponentEvents` via `Pending*` markers is a
// follow-up (Phase 3b).
//
// `SaveGltfFile` is the inverse: walks the registry + buffers and writes a glTF/.glb to disk.
//
// Sidecars below carry source-form data that GPU+registry state can't recreate verbatim.

#pragma once

#include "GltfScene.h"
#include "Image.h"
#include "entt_fwd.h"
#include "numeric/mat4.h"
#include "numeric/vec3.h"
#include "vulkan/Slots.h"

#include <array>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

struct DescriptorSlots;
struct EnvironmentStore;
struct MeshStore;
struct SceneBuffers;
struct TextureStore;

// Per-entity source-index sidecars for stable round-trip ordering / referencing. Build uses these
// rather than `SceneNode` (which has been mutated by skinning/armature re-parenting) and rather
// than entt iteration order (which doesn't track source array order).
struct SourceNodeIndex {
    uint32_t Value{};
};
struct SourceParentNodeIndex {
    uint32_t Value{};
}; // absent on scene roots
struct SourceSiblingIndex {
    uint32_t Value{};
}; // position in parent's `ChildrenNodeIndices`
struct SourceMeshIndex {
    uint32_t Value{};
};
struct SourceCameraIndex {
    uint32_t Value{};
};
struct SourceLightIndex {
    uint32_t Value{};
};
struct SourceSkinIndex {
    uint32_t Value{};
};
struct SourcePhysicsMaterialIndex {
    uint32_t Value{};
};
struct SourceCollisionFilterIndex {
    uint32_t Value{};
};
struct SourcePhysicsJointDefIndex {
    uint32_t Value{};
};

// Source mesh slot — Triangles/Lines/Points entities sharing a `SourceMeshIndex` distinguished by this.
enum class MeshKind : uint8_t { Triangles,
                                Lines,
                                Points };
struct SourceMeshKind {
    MeshKind Value{MeshKind::Triangles};
};

// Source-form names that don't survive the runtime model:
// Camera/Light: `::Camera`/`PunctualLight` are inlined on the object entity, whose `Name` is the
//   *object* name, not the definition's name in `cameras[]`/`lights[]`.
// Skin/Object: `CreateName` uniquifies via `_N` suffixes when source had collisions; we keep the raw value.
// Mesh: mesh-data entities don't appear in the object name registry.
struct CameraName {
    std::string Value;
};
struct LightName {
    std::string Value;
};
struct SkinName {
    std::string Value;
};
struct SourceObjectName {
    std::string Value;
};
struct MeshName {
    std::string Value;
};

// Source-side glTF node transform stored as a 4x4 matrix — engine uses TRS at runtime.
struct SourceMatrixTransform {
    mat4 Value{1.f};
};

// Source name was empty; populate auto-synthesizes one for usability, build skips emit.
struct SourceEmptyName {};

// Per-primitive data that `MeshStore::CreateMesh` consumes and discards. On the Triangles entity.
struct MeshSourceLayout {
    std::vector<uint32_t> VertexCounts;
    std::vector<uint32_t> AttributeFlags;
    std::vector<uint8_t> HasSourceIndices;
    std::vector<std::vector<std::optional<uint32_t>>> VariantMappings;
    uint8_t Colors0ComponentCount{};
    // target-major: target0[v0..vN], ...; `MorphTargetVertex` only carries position+normal deltas.
    std::vector<vec3> MorphTangentDeltas;
};

// Material texture slots, in the order they appear on `gltf::MaterialData`.
enum MaterialTextureSlot : uint8_t {
    MTS_BaseColor,
    MTS_MetallicRoughness,
    MTS_Normal,
    MTS_Occlusion,
    MTS_Emissive,
    MTS_Specular,
    MTS_SpecularColor,
    MTS_SheenColor,
    MTS_SheenRoughness,
    MTS_Transmission,
    MTS_DiffuseTransmission,
    MTS_DiffuseTransmissionColor,
    MTS_VolumeThickness,
    MTS_Clearcoat,
    MTS_ClearcoatRoughness,
    MTS_ClearcoatNormal,
    MTS_Anisotropy,
    MTS_Iridescence,
    MTS_IridescenceThickness,
    MTS_Count,
};

// Per-material delta of fields `gltf::FromGpu(buffers.Materials.Get(i))` can't recover:
// emissive_strength split, KHR_texture_transform meta on the 5 base textures, source texture
// indices per slot (PBRMaterial holds bindless slots), optional extension-block presence.
// Build = `FromGpu(...)` then patch from this.
struct MaterialSourceMeta {
    std::optional<float> EmissiveStrength;
    std::array<gltf::TextureTransformMeta, 5> BaseSlotMeta{}; // BaseColor..Emissive
    std::array<uint32_t, MTS_Count> TextureSlots = [] { std::array<uint32_t, MTS_Count> a; a.fill(InvalidSlot); return a; }();
    bool NameWasEmpty{};

    enum ExtensionBit : uint16_t {
        ExtIor = 1u << 0,
        ExtDispersion = 1u << 1,
        ExtEmissiveStrength = 1u << 2,
        ExtSheen = 1u << 3,
        ExtSpecular = 1u << 4,
        ExtTransmission = 1u << 5,
        ExtDiffuseTransmission = 1u << 6,
        ExtVolume = 1u << 7,
        ExtClearcoat = 1u << 8,
        ExtAnisotropy = 1u << 9,
        ExtIridescence = 1u << 10,
    };
    uint16_t ExtensionPresence{};
};

// Source-form scene-level data on the SceneEntity — encoded image bytes, sampler-config collapse,
// asset.* metadata, etc. Cameras/lights round-trip via per-entity components above.
struct GltfSourceAssets {
    std::string Copyright, Generator, MinVersion;
    std::string AssetExtras, AssetExtensions; // raw minified JSON
    std::string DefaultSceneName;
    std::vector<uint32_t> DefaultSceneRoots;
    std::vector<std::string> ExtensionsRequired;
    std::vector<std::string> MaterialVariants;
    std::unordered_map<uint64_t, std::string> ExtrasByEntity;
    std::vector<MaterialSourceMeta> MaterialMetas;
    std::vector<gltf::Texture> Textures;
    std::vector<gltf::Image> Images;
    std::vector<gltf::Sampler> Samplers;
    std::vector<std::string> AnimationOrder; // engine merges per-entity clips by name; build needs source order
    std::optional<gltf::ImageBasedLight> ImageBasedLight; // source IBL definition; runtime keeps the prefiltered cubemap
};

namespace gltf {

struct PopulateContext {
    entt::registry &R;
    entt::entity SceneEntity;
    DescriptorSlots &Slots;
    SceneBuffers &Buffers;
    MeshStore &Meshes;
    TextureStore &Textures;
    EnvironmentStore &Environments;
};

struct PopulateResult {
    entt::entity FirstMesh{null_entity};
    entt::entity Active{null_entity};
    entt::entity FirstCameraObject{null_entity};
    bool ImportedAnimation{false};
};

// Parse a glTF/.glb at `path` and drain it into the ECS / UMA buffers in one pass.
std::expected<PopulateResult, std::string>
LoadGltfFile(const std::filesystem::path &path, PopulateContext ctx);

// Walk the ECS + UMA buffers and write a glTF/.glb to `path`.
std::expected<void, std::string>
SaveGltfFile(const entt::registry &, entt::entity scene_entity, const SceneBuffers &, const MeshStore &, const std::filesystem::path &path);

} // namespace gltf
