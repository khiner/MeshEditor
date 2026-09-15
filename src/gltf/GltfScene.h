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

#include "gltf/SourceAssets.h"
#include "numeric/mat4.h"
#include "state/Entity.h"

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
// Distinguishes topology entities that share a source mesh.
enum class MeshKind : uint8_t {
    Triangles,
    Lines,
    Points
};

// Source references preserve glTF ordering and hierarchy independently of the runtime hierarchy.
// Every imported object, node stub, and bone carries one. Armature objects have no source node, so Index is empty.
struct GltfNode {
    std::optional<uint32_t> Index, Parent, Sibling, Camera, Light;
    // Retains a source matrix while runtime state uses TRS.
    std::optional<mat4> Matrix;
    // Source names that runtime naming changed. Empty when the runtime name matches.
    std::string Name, CameraName, LightName;
    // The source name was empty and the runtime name is synthesized, so saves omit it.
    bool EmptyName{};
};

// Source order of a scene, physics material, collision filter, or joint definition.
struct SourceIndex {
    uint32_t Value{};
};

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
    uint32_t Index{};
    MeshKind Kind{MeshKind::Triangles};
    std::string Name;
};

namespace gltf {
struct LoadContext {
    state::Scene &R;
    state::Entity Viewport;
    mtl::BindlessSet &Slots;
    GpuBuffers &Buffers;
    MeshStore &Meshes;
    TextureStore &Textures;
    EnvironmentStore &Environments;
};

struct LoadResult {
    state::Entity FirstCameraObject{state::Null};
    bool ImportedAnimation{false};
};

struct SaveOptions {
    uint8_t LossyImageQuality{75}; // Range 1-100; ignored for PNG.
};

// Ctx and BufCtx may be null when no image requires GPU readback.
struct SaveContext {
    const state::Scene &R;
    state::Entity Viewport;
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
void SwitchActiveScene(state::Scene &, state::Entity scene);

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
