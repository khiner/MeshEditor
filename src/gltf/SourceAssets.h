#pragma once

#include "Image.h"
#include "ImageBasedLight.h"
#include "entt_fwd.h"
#include "vulkan/Slots.h"

// Source-form glTF data preserved across round-trip but not consumed by the runtime, stored on ECS
// components. Lives apart from the loader (GltfScene.h) so consumers that only read/write these
// components don't pull in the glTF import/export machinery.

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
    std::optional<uint32_t> SamplerIndex; // Index into `SourceAssets::Samplers`
    std::optional<uint32_t> ImageIndex, WebpImageIndex, BasisuImageIndex, DdsImageIndex; // Indexes into `SourceAssets::Images` in resolution order.
    std::string Name;
};

struct Sampler {
    std::optional<Filter> MagFilter, MinFilter;
    Wrap WrapS, WrapT;
    std::string Name;
};

struct SourceScene {
    std::string Name;
    std::vector<uint32_t> RootNodeIndices;
};

// Source-form scene-level data on the viewport — encoded image bytes, sampler-config collapse, asset.* metadata, etc.
// Cameras/lights round-trip via per-entity components above.
struct SourceAssets {
    std::string Copyright, Generator, MinVersion;
    std::string AssetExtras, AssetExtensions; // raw minified JSON
    std::vector<SourceScene> Scenes;
    uint32_t ActiveSceneIndex{0}; // Becomes `asset.defaultScene` on save — the user's current view persists.
    // Per source node: bitmask of which scenes the node belongs to (bit s set ⇒ in scene s).
    // Empty / single-scene files leave this empty (everything is in the only scene).
    std::vector<uint32_t> NodeSceneMasks;
    std::vector<std::string> ExtensionsRequired;
    // Object entities (objects + armatures) created from this glTF, in load order.
    // Source of truth for selection / scene-switch — pre-existing entities aren't here, so they aren't affected.
    std::vector<entt::entity> ObjectEntities;
    std::unordered_map<uint64_t, std::string> ExtrasByEntity;
    std::vector<MaterialSourceMeta> MaterialMetas;
    std::vector<Texture> Textures;
    std::vector<Image> Images;
    std::vector<Sampler> Samplers;
    std::vector<std::string> AnimationOrder; // engine merges per-entity clips by name; build needs source order
    std::optional<ImageBasedLight> ImageBasedLight; // source IBL definition; runtime keeps the prefiltered cubemap
};
} // namespace gltf
