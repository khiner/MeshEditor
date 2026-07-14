#pragma once

#include "gltf/ImageBasedLight.h"
#include "gpu/IblSamplers.h"
#include "numeric/mat3.h"
#include "vulkan/Buffer.h"
#include "vulkan/Image.h"
#include "vulkan/VulkanResources.h"

#include <expected>
#include <filesystem>
#include <variant>

struct DescriptorSlots;
struct IblPrefilterPipelines;

namespace gltf {
struct Image;
} // namespace gltf

struct SamplerConfig {
    vk::Filter MinFilter, MagFilter;
    vk::SamplerMipmapMode MipmapMode;
    bool UsesMipmaps;
};

struct TextureEntry {
    mvk::ImageResource Image;
    vk::UniqueSampler Sampler;
    uint32_t SamplerSlot;
    uint32_t Width, Height, MipLevels;
    // Sampler build inputs, retained so the sampler can be rebuilt.
    SamplerConfig Config;
    vk::SamplerAddressMode WrapS, WrapT;
    std::string Name;
    // Index into `gltf::SourceAssets::Images` for textures materialized from a `GltfImageRef`;
    // UINT32_MAX for raw-pixel uploads (LUTs, SVG bitmaps). Used by SaveGltf for re-encode lookup.
    uint32_t SourceImageIndex{UINT32_MAX};
};

struct TextureStore {
    std::vector<TextureEntry> Textures;
    uint32_t WhiteTextureSlot;

    TextureStore() = default;
    TextureStore(const TextureStore &) = delete;
    TextureStore &operator=(const TextureStore &) = delete;
    TextureStore(TextureStore &&) = default;
    TextureStore &operator=(TextureStore &&) = default;
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
    EnvironmentPrefiltered EmptySceneWorld; // 1x1 flat-color cubemap; used when no EXT_lights_image_based asset is loaded.
    EnvironmentSelection SceneWorld, StudioWorld;

    EnvironmentStore() = default;
    EnvironmentStore(const EnvironmentStore &) = delete;
    EnvironmentStore &operator=(const EnvironmentStore &) = delete;
    EnvironmentStore(EnvironmentStore &&) = default;
    EnvironmentStore &operator=(EnvironmentStore &&) = default;
};

enum class TextureColorSpace : uint8_t {
    Srgb,
    Linear,
};

struct PendingTextureUpload {
    // Index into a `gltf::Image` vector supplied at materialization (typically
    // `gltf::SourceAssets::Images` on the viewport entity). Caller must keep the storage alive
    // until the drain pass runs.
    struct GltfImageRef {
        uint32_t ImageIndex;
    };
    struct RawPixels {
        std::vector<std::byte> Pixels;
        uint32_t Width, Height;
    };

    uint32_t SamplerSlot;
    std::variant<GltfImageRef, RawPixels> Source;
    TextureColorSpace ColorSpace;
    vk::SamplerAddressMode WrapS, WrapT;
    SamplerConfig Sampler;
    std::string Name;
};
struct PendingTextureUploads {
    std::vector<PendingTextureUpload> Items;
};

// An imported texture's upload descriptor, recording the bindless slot baked into its PBRMaterial.
// The pixel source lives in gltf::SourceAssets::Images, keyed by SourceImageIndex.
struct MaterializedTexture {
    uint32_t SamplerSlot;
    uint32_t SourceImageIndex; // index into gltf::SourceAssets::Images
    TextureColorSpace ColorSpace;
    vk::SamplerAddressMode WrapS, WrapT;
    SamplerConfig Sampler;
    std::string Name;
};
struct MaterializedTextures {
    std::vector<MaterializedTexture> Items;
};

struct PendingEnvironmentImport {
    gltf::ImageBasedLight Source;
    uint32_t DiffuseCubeSlot, SpecularCubeSlot;
};

// Tag: drain pass releases ImportedSceneWorld and resets SceneWorld back to EmptySceneWorld.
struct PendingSceneWorldClear {};

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
// Clamp a requested anisotropy to the device limit (1 when unsupported).
float ClampMaxAnisotropy(const VulkanResources &, float requested);
// Recreate all texture samplers at the given max anisotropy.
void RebuildTextureSamplers(const VulkanResources &, DescriptorSlots &, TextureStore &, float max_anisotropy);
void ReleaseCubeSamplerSlot(DescriptorSlots &, uint32_t);
void ReleaseEnvironmentSamplerSlots(DescriptorSlots &, const EnvironmentStore &);
mvk::ImageResource RenderBitmapToImage(const VulkanResources &, TextureUploadBatch &, std::span<const std::byte> data, uint32_t width, uint32_t height, vk::Format, vk::ImageSubresourceRange);

TextureEntry CreateTextureEntry(
    const VulkanResources &, TextureUploadBatch &, DescriptorSlots &,
    std::span<const std::byte> pixels, uint32_t width, uint32_t height, std::string name,
    TextureColorSpace, vk::SamplerAddressMode, vk::SamplerAddressMode, const SamplerConfig &
);
std::expected<TextureEntry, std::string> CreateTextureEntryFromEncoded(
    const VulkanResources &, TextureUploadBatch &, DescriptorSlots &,
    std::span<const std::byte>, std::string_view encoded_name, std::string texture_name,
    TextureColorSpace, vk::SamplerAddressMode, vk::SamplerAddressMode, const SamplerConfig &
);
uint32_t AllocateSamplerSlot(DescriptorSlots &);
std::pair<uint32_t, uint32_t> AllocateIblCubeSlots(DescriptorSlots &); // {diffuse, specular}

// Synchronously read an image sub-rect (mip 0, eShaderReadOnlyOptimal) into host memory
// as raw 4-byte pixels in the image's native channel order (RGBA8 or BGRA8) and row order.
std::vector<std::byte> ReadbackImageRgba8(const VulkanResources &, mvk::BufferContext &, vk::CommandPool, vk::Fence, vk::Image, vk::Offset3D, vk::Extent2D);
// Synchronously read mip 0 of an RGBA8 texture into host memory.
std::expected<std::vector<std::byte>, std::string> ReadbackTextureRgba8(const VulkanResources &, mvk::BufferContext &, vk::CommandPool, vk::Fence, const TextureEntry &);

std::expected<TextureEntry, std::string> MaterializeTextureEntry(const VulkanResources &, TextureUploadBatch &, DescriptorSlots &, const PendingTextureUpload &, const std::vector<gltf::Image> &);
std::expected<EnvironmentPrefiltered, std::string> MaterializeEnvironmentImport(const VulkanResources &, TextureUploadBatch &, DescriptorSlots &, const PendingEnvironmentImport &, const std::vector<gltf::Image> &);
EnvironmentPrefiltered CreateIblFromHdri(
    const VulkanResources &, DescriptorSlots &,
    const IblPrefilterPipelines &, const std::filesystem::path &, const std::string &,
    vk::CommandPool, vk::Fence, mvk::BufferContext &
);
// Allocate a 1x1x6 cubemap (1 mip) of the given linear color.
EnvironmentPrefiltered BuildFlatColorEnvironment(const VulkanResources &, TextureUploadBatch &, DescriptorSlots &, vec3 color, std::string name);
IblSamplers MakeIblSamplers(const EnvironmentPrefiltered &, const EnvironmentStore &);
TextureEntry CreateDefaultLutTexture(const VulkanResources &, TextureUploadBatch &, DescriptorSlots &, const std::filesystem::path &lut_path, std::string_view name);
