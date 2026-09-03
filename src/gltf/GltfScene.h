// Round-trip limitations are tested in tests/RoundtripTest.cpp.
// Lossy conversions:
// - Additional skin influence sets (JOINTS_1+, WEIGHTS_1+) are compressed at import to the top 4 weights per vertex (sorted, renormalized).
// - KHR_mesh_quantization attributes decode to FLOAT and save as FLOAT.
// - EXT_meshopt_compression buffer views decode and save uncompressed.
// - EXT_mesh_gpu_instancing retains TRS attributes only.
// - EXT_lights_image_based retains one source IBL on the default scene.
// Unsupported conversions:
// - KHR_draco_mesh_compression does not provide geometry to the importer.
// - KHR_animation_pointer channels are omitted while their static values remain.

#pragma once

#include "entt_fwd.h"
#include "gltf/SourceAssets.h"
#include "numeric/mat4.h"

#include <expected>
#include <filesystem>

namespace mtl {
struct BindlessSet;
struct BufferContext;
struct Context;
} // namespace mtl
struct EnvironmentStore;
struct MeshStore;
struct GpuBuffers;
struct TextureStore;
namespace fastgltf {
class Asset;
} // namespace fastgltf
// Source indices preserve glTF ordering and references independently of runtime hierarchy and ECS iteration.
struct SourceNodeIndex {
    uint32_t Value{};
};
struct SourceParentNodeIndex {
    uint32_t Value{};
};
struct SourceSiblingIndex {
    uint32_t Value{};
};
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

struct GltfObject {};

// Preserves source scene order during saves.
struct SourceSceneIndex {
    uint32_t Value{};
};

// Distinguishes topology entities that share a SourceMeshIndex.
enum class MeshKind : uint8_t {
    Triangles,
    Lines,
    Points
};
struct SourceMeshKind {
    MeshKind Value{MeshKind::Triangles};
};

// Retains source names that runtime naming transforms or omits.
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

// Retains a source matrix while runtime state uses TRS.
struct SourceMatrixTransform {
    mat4 Value{1.f};
};

// Marks a synthesized runtime name that must be omitted during save.
struct SourceEmptyName {};

// Retains per-primitive source layout after CreateMesh flattens primitives.
struct MeshSourceLayout {
    std::vector<uint32_t> AttributeFlags;
    std::vector<uint8_t> HasSourceIndices;
    // Materials used without a matching active variant override.
    std::vector<uint32_t> DefaultMaterials;
    // Primitive-major optional material overrides indexed by variant.
    std::vector<std::vector<std::optional<uint32_t>>> VariantMappings;
    uint8_t Colors0ComponentCount{};
    // Target-major tangent deltas omitted from MorphTargetVertex.
    std::vector<vec3> MorphTangentDeltas;
};

namespace gltf {
struct LoadContext {
    entt::registry &R;
    entt::entity Viewport;
    mtl::BindlessSet &Slots;
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
    uint8_t LossyImageQuality{75}; // Range 1-100; ignored for PNG.
};

// Ctx and BufCtx may be null when no image requires GPU readback.
struct SaveContext {
    const entt::registry &R;
    entt::entity Viewport;
    const GpuBuffers &Buffers;
    const MeshStore &Meshes;
    const TextureStore &Textures;
    const mtl::Context *Ctx{nullptr};
    mtl::BufferContext *BufCtx{nullptr};
    SaveOptions Options{};
};

std::expected<LoadResult, std::string> LoadGltf(const std::filesystem::path &, LoadContext);
std::expected<void, std::string> SaveGltf(const std::filesystem::path &, const SaveContext &);

// Parses with import extensions, loads external buffers, and decodes meshopt buffer views.
std::expected<fastgltf::Asset, std::string> ParseGltfAsset(const std::filesystem::path &);

// Activates `scene` when it names an inactive scene.
void SwitchActiveScene(entt::registry &, entt::entity scene);

// Mirrors fastgltf::Category bits used in SourceAssets::ExtrasByEntity keys.
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
