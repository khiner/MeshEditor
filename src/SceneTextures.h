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
std::expected<TextureEntry, std::string> CreateTextureEntryFromImage(
    const SceneVulkanResources &, TextureUploadBatch &, DescriptorSlots &,
    const gltf::Image &, std::string texture_name,
    TextureColorSpace, vk::SamplerAddressMode wrap_s, vk::SamplerAddressMode wrap_t,
    const SamplerConfig &
);
std::expected<EnvironmentPrefiltered, std::string> CreateIblFromExtIbl(
    const SceneVulkanResources &, TextureUploadBatch &, DescriptorSlots &,
    const std::vector<gltf::Image> &, const gltf::ImageBasedLight &
);
EnvironmentPrefiltered CreateIblFromHdri(
    const SceneVulkanResources &, DescriptorSlots &,
    const IblPrefilterPipelines &, const std::filesystem::path &, const std::string &,
    vk::CommandPool, vk::Fence, mvk::BufferContext &
);
IblSamplers MakeIblSamplers(const EnvironmentPrefiltered &, const EnvironmentStore &);
TextureEntry CreateDefaultLutTexture(
    const SceneVulkanResources &, TextureUploadBatch &, DescriptorSlots &,
    std::string_view lut_path, std::string_view name
);
