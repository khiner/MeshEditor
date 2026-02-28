#pragma once

#include "SceneVulkanResources.h"
#include "gltf/Image.h"
#include "gpu/IblSamplers.h"
#include "vulkan/Image.h"

#include <expected>
#include <filesystem>
#include <span>

struct DescriptorSlots;
struct IblPrefilterPipelines;

namespace mvk {
struct BufferContext;
} // namespace mvk

struct TextureEntry {
    mvk::ImageResource Image{};
    vk::UniqueSampler Sampler{};
    uint32_t SamplerSlot{InvalidSlot};
    uint32_t Width{0}, Height{0}, MipLevels{1};
    std::string Name;
};

struct TextureStore {
    std::vector<TextureEntry> Textures;
    uint32_t WhiteTextureSlot{InvalidSlot};
};

struct CubemapEntry {
    mvk::ImageResource Image{};
    vk::UniqueSampler Sampler{};
    uint32_t SamplerSlot{InvalidSlot};
    uint32_t Size{0}, MipLevels{1};
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
    IblSamplers Ibl{};
    std::string Name;
};

struct EnvironmentStore {
    std::vector<HdriEntry> Hdris;
    uint32_t ActiveHdriIndex{0};
    TextureEntry BrdfLut, SheenELut, CharlieLut;
    std::optional<EnvironmentPrefiltered> ImportedSceneWorld;
    EnvironmentSelection SceneWorld, StudioWorld;
};

struct SamplerConfig {
    vk::Filter MinFilter{vk::Filter::eLinear}, MagFilter{vk::Filter::eLinear};
    vk::SamplerMipmapMode MipmapMode{vk::SamplerMipmapMode::eLinear};
    bool UsesMipmaps{true};
};

enum class TextureColorSpace : uint8_t {
    Srgb,
    Linear,
};

std::vector<uint32_t> CollectSamplerSlots(std::span<const TextureEntry>);
void ReleaseSamplerSlots(DescriptorSlots &, std::span<const uint32_t>);
void ReleaseCubeSamplerSlot(DescriptorSlots &, uint32_t);
void ReleaseEnvironmentSamplerSlots(DescriptorSlots &, const EnvironmentStore &);
TextureEntry CreateTextureEntry(
    const SceneVulkanResources &, mvk::BufferContext &, vk::CommandPool, vk::Fence, DescriptorSlots &,
    std::span<const std::byte> pixels, uint32_t w, uint32_t h, std::string name,
    TextureColorSpace, vk::SamplerAddressMode, vk::SamplerAddressMode, const SamplerConfig &
);
std::expected<TextureEntry, std::string> CreateTextureEntryFromEncoded(
    const SceneVulkanResources &, mvk::BufferContext &, vk::CommandPool, vk::Fence, DescriptorSlots &,
    std::span<const std::byte>, std::string_view encoded_name, std::string texture_name,
    TextureColorSpace, vk::SamplerAddressMode, vk::SamplerAddressMode, const SamplerConfig &
);
std::expected<EnvironmentPrefiltered, std::string> CreateIblFromExtIbl(
    const SceneVulkanResources &, mvk::BufferContext &, vk::CommandPool, vk::Fence, DescriptorSlots &,
    const std::vector<gltf::Image> &, const gltf::ImageBasedLight &
);
EnvironmentPrefiltered CreateIblFromHdri(
    const SceneVulkanResources &, mvk::BufferContext &, vk::CommandPool, vk::Fence, DescriptorSlots &,
    const IblPrefilterPipelines &, const std::filesystem::path &, const std::string &
);
IblSamplers MakeIblSamplers(const EnvironmentPrefiltered &, const EnvironmentStore &);
TextureEntry CreateDefaultLutTexture(
    const SceneVulkanResources &, mvk::BufferContext &, vk::CommandPool, vk::Fence, DescriptorSlots &,
    std::string_view lut_path, std::string_view name
);
