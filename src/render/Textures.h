#pragma once

#include "gltf/ImageBasedLight.h"
#include "gpu/IblSamplers.h"
#include "metal/Buffer.h"
#include "metal/Image.h"
#include "metal/Shader.h"
#include "numeric/mat3.h"

#include <expected>
#include <filesystem>
#include <variant>

namespace mtl {
struct BindlessSet;
} // namespace mtl
struct IblPrefilterPipelines;

namespace gltf {
struct Image;
} // namespace gltf

inline constexpr float MaxSamplerAnisotropy{16.f};

struct ActiveSamplerAnisotropy {
    float Value{1.f};
};

struct SamplerConfig {
    MTL::SamplerMinMagFilter MinFilter, MagFilter;
    MTL::SamplerMipFilter MipmapMode;
    bool UsesMipmaps;
};

struct TextureEntry {
    mtl::Texture Image;
    NS::SharedPtr<MTL::SamplerState> Sampler;
    uint32_t SamplerSlot;
    // Sampler build inputs, retained so the sampler can be rebuilt.
    SamplerConfig Config;
    MTL::SamplerAddressMode WrapS, WrapT;
    std::string Name;
    // Index into `gltf::SourceAssets::Images` for textures materialized from a `GltfImageRef`.
    // UINT32_MAX denotes raw-pixel uploads such as LUTs and SVG bitmaps.
    // SaveGltf uses this value for re-encode lookup.
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
    mtl::Texture Image;
    NS::SharedPtr<MTL::SamplerState> Sampler;
    uint32_t SamplerSlot;
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
    EnvironmentPrefiltered EmptySceneWorld; // 1x1 flat-color cubemap used without an EXT_lights_image_based asset.
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
    // Indexes the glTF image array supplied at materialization.
    // The caller retains that array through the drain pass.
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
    MTL::SamplerAddressMode WrapS, WrapT;
    SamplerConfig Sampler;
    std::string Name;
};
struct PendingTextureUploads {
    std::vector<PendingTextureUpload> Items;
};

// Records an imported texture's material slot and glTF source image.
struct MaterializedTexture {
    uint32_t SamplerSlot;
    uint32_t SourceImageIndex;
    TextureColorSpace ColorSpace;
    MTL::SamplerAddressMode WrapS, WrapT;
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

struct TextureUploadBatch {
    const mtl::Context *Ctx{nullptr};
    mtl::LibraryCache *Libraries{nullptr};
    MTL::CommandBuffer *Cb{nullptr};
    NS::SharedPtr<MTL::SamplerState> MipSampler{};
    std::vector<std::pair<MTL::PixelFormat, mtl::RenderPipeline>> MipPipelines{};
};

TextureUploadBatch BeginTextureUploadBatch(const mtl::Context &, mtl::LibraryCache &);
void SubmitTextureUploadBatch(TextureUploadBatch &);

std::vector<uint32_t> CollectSamplerSlots(std::span<const TextureEntry>);
void ReleaseSamplerSlots(mtl::BindlessSet &, std::span<const uint32_t>);
// Clamp a requested anisotropy to the device limit (1 when unsupported).
float ClampMaxAnisotropy(float requested);
// Recreate all texture samplers at the given max anisotropy.
void RebuildTextureSamplers(const mtl::Context &, mtl::BindlessSet &, TextureStore &, float max_anisotropy);
void ReleaseCubeSamplerSlot(mtl::BindlessSet &, uint32_t);
void ReleaseEnvironmentSamplerSlots(mtl::BindlessSet &, const EnvironmentStore &);

TextureEntry CreateTextureEntry(
    const mtl::Context &, TextureUploadBatch &, mtl::BindlessSet &,
    std::span<const std::byte> pixels, uint32_t width, uint32_t height, std::string name,
    TextureColorSpace, MTL::SamplerAddressMode, MTL::SamplerAddressMode, const SamplerConfig &, float max_anisotropy
);
std::expected<TextureEntry, std::string> CreateTextureEntryFromEncoded(
    const mtl::Context &, TextureUploadBatch &, mtl::BindlessSet &,
    std::span<const std::byte>, std::string_view encoded_name, std::string texture_name,
    TextureColorSpace, MTL::SamplerAddressMode, MTL::SamplerAddressMode, const SamplerConfig &, float max_anisotropy
);
uint32_t AllocateSamplerSlot(mtl::BindlessSet &);
std::pair<uint32_t, uint32_t> AllocateIblCubeSlots(mtl::BindlessSet &); // {diffuse, specular}

// Synchronous mip-0 readback in the image's native RGBA8/BGRA8 order.
std::vector<std::byte> ReadbackImageRgba8(const mtl::Context &, const mtl::Texture &, uint32_t x, uint32_t y, mtl::Extent2D);
// Synchronously read mip 0 of an RGBA8 texture into host memory.
std::expected<std::vector<std::byte>, std::string> ReadbackTextureRgba8(const mtl::Context &, const TextureEntry &);

std::expected<TextureEntry, std::string> MaterializeTextureEntry(const mtl::Context &, TextureUploadBatch &, mtl::BindlessSet &, const PendingTextureUpload &, const std::vector<gltf::Image> &, float max_anisotropy);
std::expected<EnvironmentPrefiltered, std::string> MaterializeEnvironmentImport(const mtl::Context &, mtl::BindlessSet &, const PendingEnvironmentImport &, const std::vector<gltf::Image> &);
EnvironmentPrefiltered CreateIblFromHdri(
    const mtl::Context &, mtl::BindlessSet &,
    const IblPrefilterPipelines &, const std::filesystem::path &, std::string
);
// Allocate a 1x1x6 cubemap (1 mip) of the given linear color.
EnvironmentPrefiltered BuildFlatColorEnvironment(const mtl::Context &, mtl::BindlessSet &, vec3 color, std::string name);
IblSamplers MakeIblSamplers(const EnvironmentPrefiltered &, const EnvironmentStore &);
TextureEntry CreateDefaultLutTexture(const mtl::Context &, TextureUploadBatch &, mtl::BindlessSet &, const std::filesystem::path &lut_path, std::string_view name, float max_anisotropy);
