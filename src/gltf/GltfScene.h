// Round-trip gaps (see tests/RoundtripTest.cpp for the per-path exception list):
//
// Lossy:
// - Additional skin influence sets (JOINTS_1+, WEIGHTS_1+) are compressed at import to the top 4 weights per vertex (sorted, renormalized).
//  (The spec does permit this single set of 4 - see glTF 2.0 §3.7.3.1.)
// - KHR_mesh_quantization: quantized attributes decode to FLOAT at import, save always emits FLOAT
// - EXT_mesh_gpu_instancing: per-instance TRS round-trips, but custom instancing attributes (`_FOO`) beyond TRANSLATION/ROTATION/SCALE aren't retained
//
// Unsupported (neither imported nor re-emitted):
// - KHR_draco_mesh_compression, EXT_meshopt_compression: files relying on these to carry geometry
//   will load with empty/missing vertex data, or fail entirely if the extension is listed as required.
// - KHR_animation_pointer: animation channels targeting extension pointer paths are silently dropped at import (along with their samplers).
//
// Source-form fields preserved across round-trip but not consumed by the runtime live on per-entity ECS components.

#pragma once

#include "Image.h"
#include "entt_fwd.h"
#include "numeric/mat4.h"
#include "vulkan/Slots.h"

#include <array>
#include <expected>
#include <filesystem>
#include <unordered_map>
#include <vector>

struct DescriptorSlots;
struct EnvironmentStore;
struct MeshStore;
struct SceneBuffers;
struct TextureStore;
struct SceneVulkanResources;
namespace mvk {
struct BufferContext;
}

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
enum class MeshKind : uint8_t {
    Triangles,
    Lines,
    Points
};
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

// Material texture slots, in the order they appear on `MaterialSourceMeta::TextureSlots`.
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

namespace gltf {
enum class Filter : uint16_t {
    Nearest,
    Linear,
    NearestMipMapNearest,
    LinearMipMapNearest,
    NearestMipMapLinear,
    LinearMipMapLinear,
};
enum class Wrap : uint16_t {
    ClampToEdge,
    MirroredRepeat,
    Repeat,
};

// TextureInfo.TexCoord is the effective value (override if present, else base).
struct TextureTransformMeta {
    bool SourceHadExtension{};
    uint32_t SourceBaseTexCoord{};
    std::optional<uint32_t> SourceTexCoordOverride{};
};

// Per-material delta of fields `buffers.Materials` can't recover:
// emissive_strength split, KHR_texture_transform meta on the 5 base textures, source texture
// indices per slot (PBRMaterial holds bindless slots), optional extension-block presence.
// Save reads `PBRMaterial` from the GPU buffer, then uses this to gate which extension blocks
// to emit, restore source texture indices, and un-fold the EmissiveFactor *= strength split.
struct MaterialSourceMeta {
    std::optional<float> EmissiveStrength;
    std::array<TextureTransformMeta, 5> BaseSlotMeta{}; // BaseColor..Emissive
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

struct Texture {
    std::optional<uint32_t> SamplerIndex; // Index into `Scene::Samplers`
    std::optional<uint32_t> ImageIndex, WebpImageIndex, BasisuImageIndex, DdsImageIndex; // Indexes into `Scene::Images` in resolution order.
    std::string Name;
};

struct Sampler {
    std::optional<Filter> MagFilter, MinFilter;
    Wrap WrapS, WrapT;
    std::string Name;
};

// Source-form scene-level data on the SceneEntity — encoded image bytes, sampler-config collapse,
// asset.* metadata, etc. Cameras/lights round-trip via per-entity components above.
struct SourceAssets {
    std::string Copyright, Generator, MinVersion;
    std::string AssetExtras, AssetExtensions; // raw minified JSON
    std::string DefaultSceneName;
    std::vector<uint32_t> DefaultSceneRoots;
    std::vector<std::string> ExtensionsRequired;
    std::vector<std::string> MaterialVariants;
    std::unordered_map<uint64_t, std::string> ExtrasByEntity;
    std::vector<MaterialSourceMeta> MaterialMetas;
    std::vector<Texture> Textures;
    std::vector<Image> Images;
    std::vector<Sampler> Samplers;
    std::vector<std::string> AnimationOrder; // engine merges per-entity clips by name; build needs source order
    std::optional<ImageBasedLight> ImageBasedLight; // source IBL definition; runtime keeps the prefiltered cubemap
};

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
    entt::entity Active{null_entity}, FirstMesh{null_entity}, FirstCameraObject{null_entity};
    bool ImportedAnimation{false};
};

struct SaveOptions {
    uint8_t LossyImageQuality{75}; // 1–100, ignored for PNG (lossless)
};

// Vk / BufCtx are only consulted when an image is IsDirty (or external-URI fallback fires).
// For plain passthrough saves they may be null.
struct SaveContext {
    const entt::registry &R;
    entt::entity SceneEntity;
    const SceneBuffers &Buffers;
    const MeshStore &Meshes;
    const TextureStore &Textures;
    const SceneVulkanResources *Vk{nullptr};
    mvk::BufferContext *BufCtx{nullptr};
    SaveOptions Options{};
};

std::expected<PopulateResult, std::string> LoadGltf(const std::filesystem::path &, PopulateContext);
std::expected<void, std::string> SaveGltf(const SaveContext &, const std::filesystem::path &);
} // namespace gltf
