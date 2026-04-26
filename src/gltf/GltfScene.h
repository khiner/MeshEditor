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
// Source-form fields preserved across round-trip but not consumed by the runtime live on
// per-entity ECS components (see EcsScene.h sidecars). The CPU-side intermediate that the
// fastgltf↔ECS bridge uses is private to EcsScene.cpp; the public API is `gltf::LoadGltfFile`
// / `gltf::SaveGltfFile` (declared in EcsScene.h).
//
// Public types here are the small POD carriers that ECS sidecars and SceneTextures still reference
// (Image/Sampler/Texture/MaterialData/Filter/Wrap/MimeType/TextureTransformMeta + ToGpu/FromGpu).

#pragma once

#include "gpu/PBRMaterial.h"
#include "gpu/Transform.h"
#include "numeric/vec3.h"

#include <cstdint>
#include <optional>
#include <string>

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

// CPU-side material view. Each KHR_materials_* block is std::optional, set iff the source
// carried the extension. Downstream code gets the flat GPU-bindable PBRMaterial via gltf::ToGpu().
struct MaterialData {
    vec4 BaseColorFactor{1};
    vec3 EmissiveFactor{0};
    float MetallicFactor{1}, RoughnessFactor{1}, NormalScale{1}, OcclusionStrength{1};
    MaterialAlphaMode AlphaMode{MaterialAlphaMode::Opaque};
    float AlphaCutoff{0.5};
    uint32_t DoubleSided{}, Unlit{};
    TextureInfo BaseColorTexture{}, MetallicRoughnessTexture{}, NormalTexture{}, OcclusionTexture{}, EmissiveTexture{};
    // Nested extension textures (Sheen.ColorTexture etc.) don't round-trip overrides today.
    TextureTransformMeta BaseColorMeta{}, MetallicRoughnessMeta{}, NormalMeta{}, OcclusionMeta{}, EmissiveMeta{};

    std::optional<float> Ior{}, Dispersion{}, EmissiveStrength{};

    std::optional<::Sheen> Sheen{};
    std::optional<::Specular> Specular{};
    std::optional<::Transmission> Transmission{};
    std::optional<::DiffuseTransmission> DiffuseTransmission{};
    std::optional<::Volume> Volume{};
    std::optional<::Clearcoat> Clearcoat{};
    std::optional<::Anisotropy> Anisotropy{};
    std::optional<::Iridescence> Iridescence{};
};

// An absent extension and an all-default extension render identically, so it's safe
// to flatten optionals to default-initialized fields when crossing into the GPU type.
PBRMaterial ToGpu(const MaterialData &);
MaterialData FromGpu(const PBRMaterial &);

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
} // namespace gltf
