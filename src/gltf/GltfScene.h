// Round-trip limitations are tested in tests/RoundtripTest.cpp.
// Lossy conversions:
// - Additional skin influence sets (JOINTS_1+, WEIGHTS_1+) are compressed at import to the top 4 weights per vertex (sorted, renormalized).
// - KHR_mesh_quantization attributes decode to FLOAT and save as FLOAT.
// - EXT_meshopt_compression buffer views decode and save uncompressed.
// - EXT_mesh_gpu_instancing retains TRS attributes only.
// - EXT_lights_image_based retains the default scene's source IBL images, which every scene's light shares.
// - KHR_animation_pointer channels to properties outside the pointer rows are dropped.
// Unsupported conversions:
// - KHR_draco_mesh_compression does not provide geometry to the importer.

#pragma once

#include "gltf/GltfSchema.h"
#include "gltf/SourceAssets.h"
#include "gpu/MaterialPattern.h"
#include "state/Entity.h"

#include <expected>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>

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
// Every imported object, node stub, and bone carries one.
// Armature objects have no source node, so Index is empty.
struct GltfNode {
    std::optional<uint32_t> Index, Parent, Sibling, Camera, Light;
    // Retains a source matrix while runtime state uses TRS.
    std::optional<mat4> Matrix;
    // Source names that runtime naming changed.
    // Empty when the runtime name matches.
    std::string Name, CameraName, LightName;
    // The source name was empty and the runtime name is synthesized, so saves omit it.
    bool EmptyName{};
};

// Source order of a scene, physics material, collision filter, or joint description.
struct SourceIndex {
    uint32_t Value{};
};

// Retains per-primitive source layout after CreateMesh flattens primitives.
struct MeshSourceLayout {
    std::vector<uint32_t> AttributeFlags;
    std::vector<uint8_t> HasSourceIndices;
    // Materials used without a matching active variant override.
    std::vector<state::Entity> DefaultMaterials;
    // Primitive-major optional material overrides indexed by variant.
    std::vector<std::vector<std::optional<state::Entity>>> VariantMappings;
    uint8_t Colors0ComponentCount{};
    // Target-major tangent deltas omitted from MorphTargetVertex.
    std::vector<vec3> MorphTangentDeltas;
    uint32_t Index{};
    MeshKind Kind{MeshKind::Triangles};
    std::string Name;
};

// A mesh whose source layout maps some primitive's material for some variant.
struct HasVariantMappings {};

namespace gltf {
struct LoadError {
    std::string Message;
    bool UnsupportedExtensions{false};
};

struct LoadResult {
    state::Entity FirstCameraObject{state::Null};
    bool ImportedAnimation{false};
};

// An input a material's node graph computes, evaluated at each texel of a Width by Height grid: over the first texture coordinates of Mesh, or with a null Mesh over the unit square of texture coordinates, where the input's values depend on no mesh.
// Texels holds Components floats per texel, row by row from the top, in the material's units.
struct BakedInput {
    PbrInput Input;
    state::Entity Mesh;
    uint32_t Width, Height, Components;
    std::vector<float> Texels;
};
// The inputs a material's graph computes for a mesh that draws it and has texture coordinates.
// The meshes that draw a material share its inputs that depend on no mesh, and no input shares its texture with an input that holds a texture of its own, as BakeConflict tells.
struct BakedMaterial {
    state::Entity Material, Mesh;
    std::vector<std::shared_ptr<const BakedInput>> Inputs;
};
// Why the inputs a material's graph computes cannot pack into its glTF textures, or none: an input that shares a texture with a computed input holds a texture of its own.
std::optional<std::string> BakeConflict(const state::Registry &, state::Entity material, std::span<const PbrInput> computed);

struct SaveOptions {
    uint8_t LossyImageQuality{75}; // Range 1-100; ignored for PNG.
    // Each mesh's primitives drawing a material listed here for it export a material whose textures hold the baked inputs.
    // Meshes whose baked inputs are the same share one such material, and materials share the texture of the same inputs.
    std::span<const BakedMaterial> Baked{};
};

// Parses and validates the whole document before creating any entity or touching any store, so a failed load leaves the scene unchanged.
// The default scene's roots join the frame's scene, and every other source scene becomes a scene of its own.
std::expected<LoadResult, LoadError> LoadGltf(const std::filesystem::path &, state::Registry &);
// Re-encodes dirty images through the Metal context when one is registered.
std::expected<void, std::string> SaveGltf(const std::filesystem::path &, const state::Registry &, SaveOptions = {});

// Parses with import extensions, loads external buffers, and decodes meshopt buffer views.
std::expected<fastgltf::Asset, LoadError> ParseGltfAsset(const std::filesystem::path &);

// The scenes in source order, then the scenes without a source, with ties in entity index order.
std::vector<state::Entity> OrderedScenes(const state::Registry &);

// Extras keys use fastgltf::Category bits.
// These name the categories the UI reads.
constexpr uint32_t ExtrasCameras = 1u << 7, ExtrasMeshes = 1u << 9, ExtrasNodes = 1u << 11, ExtrasLights = 1u << 18;
std::optional<std::string_view> GetExtras(const SourceAssets &, uint32_t category, uint32_t source_index);
} // namespace gltf
