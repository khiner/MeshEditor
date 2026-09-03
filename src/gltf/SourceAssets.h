#pragma once

#include "Image.h"
#include "ImageBasedLight.h"
#include "metal/Slots.h"

#include <map>

// Source-form glTF data retained for round-trip serialization.

// Material texture slots in MaterialSourceMeta::TextureSlots order.
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

// TexCoord contains the extension override or base value.
struct TextureTransformMeta {
    bool SourceHadExtension{};
    uint32_t SourceBaseTexCoord{};
    std::optional<uint32_t> SourceTexCoordOverride{};
};

// Retains per-material source data unavailable from buffers.Materials.
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
    std::optional<uint32_t> SamplerIndex; // Index into `SourceAssets::Samplers`
    std::optional<uint32_t> ImageIndex, WebpImageIndex, BasisuImageIndex, DdsImageIndex; // Indexes into `SourceAssets::Images` in resolution order.
    std::string Name;
};

// Returns the first available image for a texture.
inline std::optional<uint32_t> ResolveImageIndex(const Texture &t) {
    if (t.ImageIndex) return t.ImageIndex;
    if (t.WebpImageIndex) return t.WebpImageIndex;
    if (t.BasisuImageIndex) return t.BasisuImageIndex;
    return t.DdsImageIndex;
}

struct Sampler {
    std::optional<Filter> MagFilter, MinFilter;
    Wrap WrapS, WrapT;
    std::string Name;
};

// Source-form scene data retained on the viewport.
struct SourceAssets {
    std::string Copyright, Generator, MinVersion;
    std::string AssetExtras, AssetExtensions; // Minified JSON.
    std::vector<std::string> ExtensionsRequired;
    std::map<uint64_t, std::string> ExtrasByEntity; // Ordered for deterministic snapshots.
    std::vector<MaterialSourceMeta> MaterialMetas;
    std::vector<Texture> Textures;
    std::vector<Image> Images;
    std::vector<Sampler> Samplers;
    std::vector<std::string> AnimationOrder; // Source animation order.
    std::optional<ImageBasedLight> ImageBasedLight; // Source IBL definition.
};
} // namespace gltf

// Canonical source assets required for re-export and texture or IBL restoration.
