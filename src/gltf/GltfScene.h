// Round-trip gaps (see tests/RoundtripTest.cpp for the per-path exception list):
//
// Lossy:
// - Additional skin influence sets (JOINTS_1+, WEIGHTS_1+) are compressed at import to the top 4 weights per vertex (sorted, renormalized).
//  (The spec does permit this single set of 4 - see glTF 2.0 §3.7.3.1.)
// - KHR_mesh_quantization: quantized attributes decode to FLOAT at import, save always emits FLOAT
// - EXT_meshopt_compression: compressed bufferViews decode to plain data at import, save always emits uncompressed
// - EXT_mesh_gpu_instancing: per-instance TRS round-trips, but custom instancing attributes (`_FOO`) beyond TRANSLATION/ROTATION/SCALE aren't retained
// - EXT_lights_image_based: per-scene IBL assignments collapse to a single source IBL on the default scene
//
// Unsupported (neither imported nor re-emitted):
// - KHR_draco_mesh_compression: files relying on this to carry geometry
//   will load with empty/missing vertex data, or fail entirely if the extension is listed as required.
// - KHR_animation_pointer: animation channels targeting extension pointer paths are silently dropped at import (along with their samplers).
//   The static value the pointer targets still round-trips. E.g. KHR_node_visibility per-node `visible` flags persist - only their animation is lost.
//
// Source-form fields preserved across round-trip but not consumed by the runtime live on per-entity ECS components.

#pragma once

#include "entt_fwd.h"
#include "gltf/SourceAssets.h"
#include "numeric/mat4.h"

#include <expected>
#include <filesystem>

struct DescriptorSlots;
struct EnvironmentStore;
struct MeshStore;
struct GpuBuffers;
struct TextureStore;
struct VulkanResources;
namespace mvk {
struct BufferContext;
} // namespace mvk

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
struct SourcePhysicsMaterialIndex {
    uint32_t Value{};
};
struct SourceCollisionFilterIndex {
    uint32_t Value{};
};
struct SourcePhysicsJointDefIndex {
    uint32_t Value{};
};

// If present, object was created by a glTF import.
struct GltfObject {};

// A `Scene`'s index in the source `scenes` array, kept to preserve scene order on save.
struct SourceSceneIndex {
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

// Edit-stable glTF provenance not re-derivable from runtime state.
// Covers round-trip ordering, referencing, and source-form names and transforms.

// Source per-primitive layout retained on the Triangles entity after `CreateMesh` flattens primitives.
// Drives runtime material-variant resolution, carries morph tangent deltas the arena doesn't store, and preserves the structure for faithful glTF re-export.
struct MeshSourceLayout {
    std::vector<uint32_t> VertexCounts;
    std::vector<uint32_t> AttributeFlags;
    std::vector<uint8_t> HasSourceIndices;
    // No KHR_materials_variants variant is active, or the active variant doesn't override this primitive.
    std::vector<uint32_t> DefaultMaterials;
    // Outer indexed by primitive. inner size == variant count (empty when primitive has no mappings).
    // nullopt means variant has no override (falls back to DefaultMaterials).
    std::vector<std::vector<std::optional<uint32_t>>> VariantMappings;
    uint8_t Colors0ComponentCount{};
    // target-major: target0[v0..vN], ...; `MorphTargetVertex` only carries position+normal deltas.
    std::vector<vec3> MorphTangentDeltas;
};

namespace gltf {
struct LoadContext {
    entt::registry &R;
    entt::entity Viewport;
    DescriptorSlots &Slots;
    GpuBuffers &Buffers;
    MeshStore &Meshes;
    TextureStore &Textures;
    EnvironmentStore &Environments;
};

struct LoadResult {
    entt::entity FirstCameraObject{null_entity};
    bool ImportedAnimation{false};
};

struct SaveOptions {
    uint8_t LossyImageQuality{75}; // 1–100, ignored for PNG (lossless)
};

// Vk / BufCtx are only consulted when an image is IsDirty (or external-URI fallback fires).
// For plain passthrough saves they may be null.
struct SaveContext {
    const entt::registry &R;
    entt::entity Viewport;
    const GpuBuffers &Buffers;
    const MeshStore &Meshes;
    const TextureStore &Textures;
    const VulkanResources *Vk{nullptr};
    mvk::BufferContext *BufCtx{nullptr};
    SaveOptions Options{};
};

std::expected<LoadResult, std::string> LoadGltf(const std::filesystem::path &, LoadContext);
std::expected<void, std::string> SaveGltf(const std::filesystem::path &, const SaveContext &);

// Make `scene` the active scene shown in the viewport. No-op if it's already active or not a scene.
void SwitchActiveScene(entt::registry &, entt::entity scene);

// Mirrors fastgltf::Category bit values (asserted in .cpp). Used as the category half of
// `SourceAssets::ExtrasByEntity` keys, which the loader writes via fastgltf's parse callback.
enum class ExtrasCategory : uint32_t {
    Images = 1u << 3,
    Samplers = 1u << 4,
    Textures = 1u << 5,
    Animations = 1u << 6,
    Cameras = 1u << 7,
    Materials = 1u << 8,
    Meshes = 1u << 9,
    Skins = 1u << 10,
    Nodes = 1u << 11,
    Scenes = 1u << 12,
    Lights = 1u << 18, // KHR_lights_punctual; not a top-level glTF category but identifies lights in the extras callback.
    ImageBasedLights = 1u << 19, // EXT_lights_image_based.
};
std::optional<std::string_view> GetExtras(const SourceAssets &, ExtrasCategory, uint32_t source_index);
} // namespace gltf
