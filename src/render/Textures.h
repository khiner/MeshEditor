#pragma once

#include "gltf/ImageBasedLight.h"
#include "gpu/IblSamplers.h"
#include "metal/Command.h"
#include "metal/Image.h"
#include "metal/Mipmaps.h"

#include "state/Entity.h"
#include <expected>
#include <filesystem>
#include <optional>
#include <variant>

namespace mtl {
struct BindlessSet;
} // namespace mtl
struct Pipelines;

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
    bool operator==(const SamplerConfig &) const = default;
};

enum class TextureColorSpace : uint8_t {
    Srgb,
    Linear,
};

// Sampling and naming inputs shared by an upload request, its texture record, and its restore manifest.
struct TextureParams {
    TextureColorSpace ColorSpace;
    MTL::SamplerAddressMode WrapS, WrapT;
    SamplerConfig Sampler;
    std::string Name;
    bool operator==(const TextureParams &) const = default;
};

struct TextureEntry {
    mtl::Texture Image;
    NS::SharedPtr<MTL::SamplerState> Sampler;
    uint32_t SamplerSlot;
    TextureParams Params;
    // Index into `gltf::SourceAssets::Images` for textures materialized from a `GltfImageRef`.
    // UINT32_MAX denotes the raw-pixel uploads that outlive documents: the default white texture and the LUTs.
    // SaveGltf uses this value for re-encode lookup.
    uint32_t SourceImageIndex{UINT32_MAX};
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
    TextureParams Params;
};

struct TextureStore {
    std::vector<TextureEntry> Textures;
    uint32_t WhiteTextureSlot;
    // Uploads the next event pass materializes.
    std::vector<PendingTextureUpload> PendingUploads;

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

struct PendingEnvironmentImport {
    gltf::ImageBasedLight Source;
    uint32_t DiffuseCubeSlot, SpecularCubeSlot;
};

struct EnvironmentStore {
    std::vector<HdriEntry> Hdris;
    // Sampler slots of the raw-pixel LUT textures that TextureStore materializes and owns.
    uint32_t BrdfLutSlot, SheenELutSlot, CharlieLutSlot;
    std::optional<EnvironmentPrefiltered> ImportedSceneWorld;
    EnvironmentPrefiltered EmptySceneWorld; // 1x1 flat-color cubemap used without an EXT_lights_image_based asset.
    EnvironmentSelection SceneWorld;
    // An EXT_lights_image_based import the next event pass materializes.
    std::optional<PendingEnvironmentImport> PendingImport;
    // Release the imported scene world on the next event pass.
    bool ClearRequested{};

    EnvironmentStore() = default;
    EnvironmentStore(const EnvironmentStore &) = delete;
    EnvironmentStore &operator=(const EnvironmentStore &) = delete;
    EnvironmentStore(EnvironmentStore &&) = default;
    EnvironmentStore &operator=(EnvironmentStore &&) = default;
};

// Records an imported texture's material slot and glTF source image.
struct MaterializedTexture {
    uint32_t SamplerSlot;
    uint32_t SourceImageIndex;
    TextureParams Params;
};
struct MaterializedTextures {
    std::vector<MaterializedTexture> Items;
};

// One command carries every mip generation of a materialization pass.
struct TextureUploadBatch {
    TextureUploadBatch(const mtl::Context &, const mtl::ComputePipeline &downsample);
    ~TextureUploadBatch();
    TextureUploadBatch(const TextureUploadBatch &) = delete;
    TextureUploadBatch &operator=(const TextureUploadBatch &) = delete;
    // Records the texture's mip generation, which the submit completes.
    void GenerateMipmaps(const mtl::Texture &);
    const mtl::ComputePipeline &Downsample;
    std::optional<mtl::Command> Cmd;
    NS::SharedPtr<MTL4::ComputeCommandEncoder> Encoder;
    std::vector<mtl::MipChain> Chains;
};

void SubmitTextureUploadBatch(TextureUploadBatch &);

void ReleaseTextureSlots(mtl::BindlessSet &, std::span<const TextureEntry>);
// Clamp a requested anisotropy to the device limit (1 when unsupported).
float ClampMaxAnisotropy(float requested);
// Recreate all texture samplers at the given max anisotropy.
void RebuildTextureSamplers(const mtl::Context &, mtl::BindlessSet &, TextureStore &, float max_anisotropy);
void ReleaseCubeSamplerSlot(mtl::BindlessSet &, uint32_t);
void ReleaseEnvironmentSamplerSlots(mtl::BindlessSet &, const EnvironmentStore &);

struct Rgba8Pixels {
    std::span<const std::byte> Pixels;
    uint32_t Width, Height;
};
struct MipUpload {
    uint32_t Level;
    size_t Offset, Bytes;
    uint32_t BytesPerRow;
};
// Transcoded rows of every level, each located in `Data` by its MipUpload.
struct Ktx2Pixels {
    MTL::PixelFormat Format;
    uint32_t Width, Height;
    std::span<const std::byte> Data;
    std::span<const MipUpload> Mips;
};
using TexturePixels = std::variant<Rgba8Pixels, Ktx2Pixels>;

// Uploads the pixels, generates the mip chain of a mipmapped RGBA8 image, and binds the sampler at `sampler_slot`.
TextureEntry CreateTextureEntry(const mtl::Context &, TextureUploadBatch &, mtl::BindlessSet &, uint32_t sampler_slot, const TexturePixels &, TextureParams, float max_anisotropy);
uint32_t AllocateSamplerSlot(mtl::BindlessSet &);
std::pair<uint32_t, uint32_t> AllocateIblCubeSlots(mtl::BindlessSet &); // {diffuse, specular}

// Synchronous mip-0 readback in the image's native RGBA8/BGRA8 order.
std::vector<std::byte> ReadbackImageRgba8(const mtl::Context &, const mtl::Texture &, uint32_t x, uint32_t y, mtl::Extent2D);
// Synchronously read mip 0 of an RGBA8 texture into host memory.
std::expected<std::vector<std::byte>, std::string> ReadbackTextureRgba8(const mtl::Context &, const TextureEntry &);

// The host pixels of a pending upload: RGBA8 rows, or every transcoded KTX2 level located in `Data` by its MipUpload.
struct DecodedTexture {
    std::vector<std::byte> Data;
    std::vector<MipUpload> Mips;
    MTL::PixelFormat Format{};
    uint32_t Width{}, Height{};
    uint32_t SourceImageIndex{UINT32_MAX};
};
// Reads, decodes or transcodes the upload's source on the calling thread, taking raw pixels out of the upload.
// Independent uploads decode concurrently.
std::expected<DecodedTexture, std::string> DecodeTextureSource(const state::Registry &, PendingTextureUpload &, const std::vector<gltf::Image> &);
// Uploads the decoded pixels and binds the sampler at the upload's slot.
TextureEntry MaterializeTextureEntry(const mtl::Context &, TextureUploadBatch &, mtl::BindlessSet &, const PendingTextureUpload &, const DecodedTexture &, float max_anisotropy);
std::expected<EnvironmentPrefiltered, std::string> MaterializeEnvironmentImport(const state::Registry &, mtl::BindlessSet &, const PendingEnvironmentImport &, const std::vector<gltf::Image> &);
void ResetImportedEnvironment(state::Registry &);
// Releases the imported and pending scene worlds and selects the empty one.
void Reset(EnvironmentStore &, state::Registry &);
// Release imported GPU textures while retaining the default white texture.
void ReleaseImportedTextures(state::Registry &);
// Releases the imported textures and their pending uploads, keeping the raw-pixel textures the engine owns.
void Reset(TextureStore &, state::Registry &);
// Release the imported GPU textures that no manifest item names by slot, source image and parameters.
void ReleaseUnlistedTextures(state::Registry &, std::span<const MaterializedTexture>);
// Release imported textures and reset to the default material.
void ResetImportedTexturesAndMaterials(state::Registry &);
EnvironmentPrefiltered CreateIblFromHdri(const mtl::Context &, mtl::BindlessSet &, const Pipelines &, const std::filesystem::path &, std::string);
// Allocate a 1x1x6 cubemap (1 mip) of the given linear color.
EnvironmentPrefiltered BuildFlatColorEnvironment(const mtl::Context &, mtl::BindlessSet &, vec3 color, std::string name);
IblSamplers MakeIblSamplers(const EnvironmentPrefiltered &, const EnvironmentStore &);
// Decodes a LUT image and queues it as a permanent raw-pixel upload, returning its sampler slot.
uint32_t QueueLutTexture(TextureStore &, mtl::BindlessSet &, const std::filesystem::path &lut_path, std::string name);

// The samplers of the named studio HDRI, or of the first HDRI when none has the name, prefiltering it on first use.
// Without studio HDRIs, the scene world's samplers.
IblSamplers StudioIbl(state::Registry &, std::string_view name);
// Releases every prefiltered studio HDRI, which the next frame that shows one prefilters again.
void RebuildStudioEnvironments(state::Registry &);
