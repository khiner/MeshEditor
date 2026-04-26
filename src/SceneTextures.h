#pragma once

#include "SceneVulkanResources.h"
#include "gltf/Image.h"
#include "gpu/IblSamplers.h"
#include "numeric/mat3.h"
#include "vulkan/Buffer.h"
#include "vulkan/Image.h"

#include <expected>
#include <filesystem>
#include <span>

struct DescriptorSlots;
struct IblPrefilterPipelines;

// If `TextureEntry` exists, texture is fully materialized (GPU image + sampler + descriptor written).
// In-flight textures live as `PendingTextureUpload` markers on the scene entity until the
// drain pass in `Scene::ProcessComponentEvents` materializes them.
struct TextureEntry {
    mvk::ImageResource Image;
    vk::UniqueSampler Sampler;
    uint32_t SamplerSlot;
    uint32_t Width, Height, MipLevels;
    std::string Name;
};

struct TextureStore {
    std::vector<TextureEntry> Textures;
    uint32_t WhiteTextureSlot;
};

struct CubemapEntry {
    mvk::ImageResource Image;
    vk::UniqueSampler Sampler;
    uint32_t SamplerSlot;
    uint32_t Size, MipLevels;
    std::string Name;
};

struct EnvironmentPrefiltered {
    CubemapEntry DiffuseEnv; // 32×32, 1 mip
    CubemapEntry SpecularEnv; // 256×256, 9 mips (sheen reuses this)
    std::string Name;
};

struct HdriEntry {
    std::string Name;
    std::filesystem::path Path;
    std::optional<EnvironmentPrefiltered> Prefiltered;
};

struct EnvironmentSelection {
    IblSamplers Ibl;
    std::string Name;
};

struct EnvironmentStore {
    std::vector<HdriEntry> Hdris;
    uint32_t ActiveHdriIndex;
    TextureEntry BrdfLut, SheenELut, CharlieLut;
    std::optional<EnvironmentPrefiltered> ImportedSceneWorld;
    mat3 SceneWorldRotation{1.f}; // From EXT_lights_image_based rotation quaternion.
    EnvironmentSelection SceneWorld, StudioWorld;
};

struct SamplerConfig {
    vk::Filter MinFilter, MagFilter;
    vk::SamplerMipmapMode MipmapMode;
    bool UsesMipmaps;
};

enum class TextureColorSpace : uint8_t {
    Srgb,
    Linear,
};

// Deferred-upload markers for the glTF load path. Slots in these are pre-allocated at load
// time (CPU bookkeeping), the GPU image/sampler/descriptor work runs in the drain pass.
//
// Source image bytes are referenced by index into `GltfSourceAssets::Images` on the scene
// entity rather than copied. `GltfSourceAssets` is emplaced before any pending markers are
// pushed (`gltf::EcsScene.cpp`), so the storage outlives the pending markers under the
// invariant that the drain pass runs before the next load swaps `GltfSourceAssets`.
struct PendingTextureUpload {
    uint32_t SamplerSlot;
    uint32_t SourceImageIndex;
    TextureColorSpace ColorSpace;
    vk::SamplerAddressMode WrapS, WrapT;
    SamplerConfig Sampler;
    std::string Name;
};
struct PendingTextureUploads {
    std::vector<PendingTextureUpload> Items;
};

struct PendingEnvironmentImport {
    gltf::ImageBasedLight Source;
    uint32_t DiffuseCubeSlot, SpecularCubeSlot;
};

struct StagingAlloc {
    vk::Buffer Buffer;
    vk::DeviceSize Offset;
};

struct TextureUploadBatch {
    vk::UniqueCommandBuffer Cb;
    mvk::BufferContext *Ctx;
    std::vector<mvk::Buffer> StagingChunks;
    vk::DeviceSize ChunkUsed{0};
};

TextureUploadBatch BeginTextureUploadBatch(vk::Device, vk::CommandPool, mvk::BufferContext &);
void SubmitTextureUploadBatch(TextureUploadBatch &, vk::Queue, vk::Fence, vk::Device);
StagingAlloc AllocStaging(TextureUploadBatch &, std::span<const std::byte>);

std::vector<uint32_t> CollectSamplerSlots(std::span<const TextureEntry>);
void ReleaseSamplerSlots(DescriptorSlots &, std::span<const uint32_t>);
void ReleaseCubeSamplerSlot(DescriptorSlots &, uint32_t);
void ReleaseEnvironmentSamplerSlots(DescriptorSlots &, const EnvironmentStore &);
mvk::ImageResource RenderBitmapToImage(
    const SceneVulkanResources &, TextureUploadBatch &,
    std::span<const std::byte> data, uint32_t width, uint32_t height, vk::Format, vk::ImageSubresourceRange
);
TextureEntry CreateTextureEntry(
    const SceneVulkanResources &, TextureUploadBatch &, DescriptorSlots &,
    std::span<const std::byte> pixels, uint32_t width, uint32_t height, std::string name,
    TextureColorSpace, vk::SamplerAddressMode, vk::SamplerAddressMode, const SamplerConfig &
);
std::expected<TextureEntry, std::string> CreateTextureEntryFromEncoded(
    const SceneVulkanResources &, TextureUploadBatch &, DescriptorSlots &,
    std::span<const std::byte>, std::string_view encoded_name, std::string texture_name,
    TextureColorSpace, vk::SamplerAddressMode, vk::SamplerAddressMode, const SamplerConfig &
);
uint32_t AllocateSamplerSlot(DescriptorSlots &);
std::pair<uint32_t, uint32_t> AllocateIblCubeSlots(DescriptorSlots &); // {diffuse, specular}

// Materializes a `TextureEntry` into a slot pre-allocated by `AllocateSamplerSlot`.
// `source` is the encoded image; typically resolved from `GltfSourceAssets::Images[item.SourceImageIndex]`.
std::expected<TextureEntry, std::string> MaterializeTextureEntry(
    const SceneVulkanResources &, TextureUploadBatch &, DescriptorSlots &,
    const PendingTextureUpload &item, const gltf::Image &source
);
// Materializes an `EnvironmentPrefiltered` into the two cube slots pre-allocated by `AllocateIblCubeSlots`.
std::expected<EnvironmentPrefiltered, std::string> MaterializeEnvironmentImport(
    const SceneVulkanResources &, TextureUploadBatch &, DescriptorSlots &,
    const PendingEnvironmentImport &, const std::vector<gltf::Image> &images
);
EnvironmentPrefiltered CreateIblFromHdri(
    const SceneVulkanResources &, DescriptorSlots &,
    const IblPrefilterPipelines &, const std::filesystem::path &, const std::string &,
    vk::CommandPool, vk::Fence, mvk::BufferContext &
);
IblSamplers MakeIblSamplers(const EnvironmentPrefiltered &, const EnvironmentStore &);
TextureEntry CreateDefaultLutTexture(
    const SceneVulkanResources &, TextureUploadBatch &, DescriptorSlots &,
    const std::filesystem::path &lut_path, std::string_view name
);
