#include "render/Textures.h"
#include "File.h"
#include "gltf/Image.h"
#include "image/ImageDecode.h"
#include "mesh/MeshStore.h"
#include "metal/Bindless.h"
#include "metal/RenderTarget.h"
#include "render/GpuBuffers.h"
#include "render/IblPrefilterPipelines.h"
#include "render/MaterialComponents.h"
#include "render/MaterialImport.h"
#include "render/TextureRefs.h"

#include <basisu_transcoder.h>
#include <entt/entity/registry.hpp>

#include <algorithm>
#include <array>
#include <iostream>
#include <unordered_map>

namespace {
NS::SharedPtr<MTL::SamplerState> MakeLinearSampler(const mtl::Context &ctx, MTL::SamplerAddressMode address_mode) {
    return mtl::CreateSampler(ctx, MTL::SamplerMinMagFilterLinear, MTL::SamplerMipFilterLinear, address_mode);
}

NS::SharedPtr<MTL::SamplerState> MakeSampler(
    const mtl::Context &ctx, const SamplerConfig &cfg, MTL::SamplerAddressMode wrap_s, MTL::SamplerAddressMode wrap_t, float max_anisotropy
) {
    // Anisotropic filtering only applies with a mip chain.
    const bool anisotropic = cfg.UsesMipmaps && max_anisotropy > 1.f;
    return mtl::CreateSampler(ctx, {
                                       cfg.MinFilter,
                                       cfg.MagFilter,
                                       cfg.MipmapMode,
                                       wrap_s,
                                       wrap_t,
                                       MTL::SamplerAddressModeRepeat,
                                       anisotropic ? max_anisotropy : 1.f,
                                   });
}

// Render each mip from the preceding level through linear filtering.
void GenerateMipChain(TextureUploadBatch &batch, const mtl::Texture &image, MTL::PixelFormat format) {
    if (image.MipLevels <= 1) return;
    if (!batch.MipSampler) {
        batch.MipSampler = mtl::CreateSampler(
            *batch.Ctx, MTL::SamplerMinMagFilterLinear, MTL::SamplerMipFilterNotMipmapped, MTL::SamplerAddressModeClampToEdge
        );
    }
    // The destination level is the render target, so each texture format needs its own pipeline.
    const auto targets_format = [format](const auto &entry) { return entry.first == format; };
    if (std::ranges::none_of(batch.MipPipelines, targets_format)) {
        batch.MipPipelines.emplace_back(format, mtl::RenderPipeline{*batch.Libraries, mtl::FunctionRef{"TexQuad.metal", "TexQuadVertex"}, mtl::FunctionRef{"MipDownsample.metal", "MipDownsampleFragment"}, mtl::PassFormats{.Color = {format}}});
    }
    const auto &state = std::ranges::find_if(batch.MipPipelines, targets_format)->second;
    for (uint32_t mip = 1; mip < image.MipLevels; ++mip) {
        const std::array colors{mtl::ColorAttachment{image.Handle.get(), MTL::LoadActionDontCare, MTL::StoreActionStore, {}, mip}};
        const auto pass = mtl::MakePassDescriptor(colors);
        auto *encoder = batch.Cb->renderCommandEncoder(pass);
        state.Bind(encoder);
        const auto source = mtl::CreateMipView(image, mip - 1);
        encoder->setFragmentTexture(source.get(), 0);
        encoder->setFragmentSamplerState(batch.MipSampler.get(), 0);
        encoder->drawPrimitives(MTL::PrimitiveTypeTriangleStrip, NS::UInteger(0), NS::UInteger(4));
        encoder->endEncoding();
    }
}

MTL::PixelFormat ToTextureFormat(TextureColorSpace color_space) {
    return color_space == TextureColorSpace::Srgb ? MTL::PixelFormatRGBA8Unorm_sRGB : MTL::PixelFormatRGBA8Unorm;
}

vec3 CubemapFaceDirection(uint32_t face, float u, float v) {
    switch (face) {
        case 0: return numeric::Normalize(vec3{1.f, -v, -u}); // +X
        case 1: return numeric::Normalize(vec3{-1.f, -v, u}); // -X
        case 2: return numeric::Normalize(vec3{u, 1.f, v}); // +Y
        case 3: return numeric::Normalize(vec3{u, -1.f, -v}); // -Y
        case 4: return numeric::Normalize(vec3{u, -v, 1.f}); // +Z
        default: return numeric::Normalize(vec3{-u, -v, -1.f}); // -Z
    }
}

// EXT_lights_image_based Appendix B (Romain Guy) irradiance reconstruction constants.
vec3 EvaluateIrradianceSH(const std::array<vec3, 9> &l, vec3 n) {
    static constexpr float c0{0.886227f}, c1{1.023327f}, c2{0.858086f}, c3{0.247708f}, c4{0.429043f};
    const vec3 irradiance =
        c0 * l[0] -
        c1 * n.y * l[1] +
        c1 * n.z * l[2] -
        c1 * n.x * l[3] +
        c2 * n.x * n.y * l[4] -
        c2 * n.y * n.z * l[5] +
        c3 * (3.f * n.z * n.z - 1.f) * l[6] -
        c2 * n.x * n.z * l[7] +
        c4 * (n.x * n.x - n.y * n.y) * l[8];
    return numeric::Max(irradiance, vec3{0});
}

using CubemapMipFacesF32 = std::array<DecodedImageF32, 6>;

CubemapMipFacesF32 BuildDiffuseCubemapFromIrradiance(const std::array<vec3, 9> &coefficients, uint32_t size = 32u) {
    CubemapMipFacesF32 mip{};
    for (uint32_t face = 0; face < 6u; ++face) {
        auto &image = mip[face];
        image.Width = size;
        image.Height = size;
        image.Pixels.resize(size * size * 4, 1.f);
        for (uint32_t y = 0; y < size; ++y) {
            for (uint32_t x = 0; x < size; ++x) {
                const auto u = 2.f * (x + 0.5f) / float(size) - 1.f;
                const auto v = 2.f * (y + 0.5f) / float(size) - 1.f;
                const auto rgb = EvaluateIrradianceSH(coefficients, CubemapFaceDirection(face, u, v));
                const auto offset = (size_t(y) * size + x) * 4u;
                image.Pixels[offset + 0] = rgb.x;
                image.Pixels[offset + 1] = rgb.y;
                image.Pixels[offset + 2] = rgb.z;
                image.Pixels[offset + 3] = 1.f;
            }
        }
    }
    return mip;
}

std::expected<CubemapEntry, std::string> CreateCubemapEntryFromMipFacesF32(
    const mtl::Context &ctx, mtl::BindlessSet &slots,
    uint32_t pre_allocated_slot,
    const std::vector<CubemapMipFacesF32> &mip_faces,
    std::string name
) {
    if (mip_faces.empty()) return std::unexpected{"Cubemap has no mip levels."};

    const uint32_t base_size = mip_faces.front()[0].Width;
    if (base_size == 0u || mip_faces.front()[0].Height != base_size) return std::unexpected{"Cubemap base face dimensions must be square and non-zero."};

    for (uint32_t mip = 0; mip < mip_faces.size(); ++mip) {
        const uint32_t expected = std::max(1u, base_size >> mip);
        for (uint32_t face = 0; face < 6u; ++face) {
            const auto &image = mip_faces[mip][face];
            if (image.Width != expected || image.Height != expected) {
                return std::unexpected{std::format("Cubemap mip {} face {} has size {}x{}; expected {}x{}.", mip, face, image.Width, image.Height, expected, expected)};
            }
            if (image.Pixels.size() != size_t(expected) * expected * 4u) {
                return std::unexpected{std::format("Cubemap mip {} face {} has invalid RGBA float payload size {}.", mip, face, image.Pixels.size())};
            }
        }
    }

    constexpr auto format = MTL::PixelFormatRGBA32Float;
    auto image = mtl::CreateTextureCube(ctx, format, base_size, MTL::TextureUsageShaderRead, uint32_t(mip_faces.size()));

    for (uint32_t mip = 0; mip < mip_faces.size(); ++mip) {
        const uint32_t size = std::max(1u, base_size >> mip);
        for (uint32_t face = 0; face < 6u; ++face) {
            const auto &src = mip_faces[mip][face].Pixels;
            mtl::Upload(image, mip, as_bytes(std::span<const float>{src}), size * 4u * sizeof(float), face);
        }
    }

    auto sampler = MakeLinearSampler(ctx, MTL::SamplerAddressModeClampToEdge);
    slots.SetSampler({SlotType::CubeSampler, pre_allocated_slot}, *image, sampler.get());
    return CubemapEntry{.Image = std::move(image), .Sampler = std::move(sampler), .SamplerSlot = pre_allocated_slot, .Name = std::move(name)};
}
struct MipUpload {
    uint32_t Level;
    size_t Offset, Bytes;
    uint32_t BytesPerRow;
};

struct KtxFormatPair {
    MTL::PixelFormat Format;
    basist::transcoder_texture_format BasisFmt;
};

KtxFormatPair SelectKtx2Format(const mtl::Context &ctx, TextureColorSpace cs) {
    const bool srgb = cs == TextureColorSpace::Srgb;
    if (ctx.Device->supportsBCTextureCompression()) {
        return {srgb ? MTL::PixelFormatBC7_RGBAUnorm_sRGB : MTL::PixelFormatBC7_RGBAUnorm, basist::transcoder_texture_format::cTFBC7_RGBA};
    }
    return {srgb ? MTL::PixelFormatRGBA8Unorm_sRGB : MTL::PixelFormatRGBA8Unorm, basist::transcoder_texture_format::cTFRGBA32};
}

TextureEntry CreateCompressedTextureEntry(
    const mtl::Context &ctx, mtl::BindlessSet &slots,
    uint32_t pre_allocated_slot,
    std::span<const std::byte> all_mip_data,
    std::span<const MipUpload> mips,
    MTL::PixelFormat format, uint32_t width, uint32_t height, uint32_t mip_levels,
    std::string name,
    MTL::SamplerAddressMode wrap_s, MTL::SamplerAddressMode wrap_t, const SamplerConfig &sampler_cfg, float max_anisotropy
) {
    auto image = mtl::CreateTexture2D(ctx, format, {width, height}, MTL::TextureUsageShaderRead, mip_levels);
    for (const auto &mip : mips) {
        mtl::Upload(image, mip.Level, all_mip_data.subspan(mip.Offset, mip.Bytes), mip.BytesPerRow);
    }

    auto sampler = MakeSampler(ctx, sampler_cfg, wrap_s, wrap_t, max_anisotropy);
    slots.SetSampler({SlotType::Sampler, pre_allocated_slot}, *image, sampler.get());
    return {.Image = std::move(image), .Sampler = std::move(sampler), .SamplerSlot = pre_allocated_slot, .Config = sampler_cfg, .WrapS = wrap_s, .WrapT = wrap_t, .Name = std::move(name)};
}
} // namespace

TextureUploadBatch BeginTextureUploadBatch(const mtl::Context &ctx, mtl::LibraryCache &libraries) {
    return {.Ctx = &ctx, .Libraries = &libraries, .Cb = ctx.Queue->commandBuffer()};
}

void SubmitTextureUploadBatch(TextureUploadBatch &batch) {
    if (!batch.Cb) return;
    batch.Cb->commit();
    // Materialization reads back and binds immediately after, so the batch settles before returning.
    batch.Cb->waitUntilCompleted();
    batch.Cb = nullptr;
}

std::vector<uint32_t> CollectSamplerSlots(std::span<const TextureEntry> textures) {
    std::vector<uint32_t> sampler_slots;
    sampler_slots.reserve(textures.size());
    for (const auto &texture : textures) {
        if (texture.SamplerSlot != InvalidSlot) sampler_slots.emplace_back(texture.SamplerSlot);
    }
    return sampler_slots;
}

void ReleaseSamplerSlots(mtl::BindlessSet &slots, std::span<const uint32_t> sampler_slots) {
    for (const auto sampler_slot : sampler_slots) slots.Release({SlotType::Sampler, sampler_slot});
}

float ClampMaxAnisotropy(float requested) { return std::clamp(requested, 1.f, MaxSamplerAnisotropy); }

void RebuildTextureSamplers(const mtl::Context &ctx, mtl::BindlessSet &slots, TextureStore &textures, float max_anisotropy) {
    for (auto &entry : textures.Textures) {
        entry.Sampler = MakeSampler(ctx, entry.Config, entry.WrapS, entry.WrapT, max_anisotropy);
        slots.SetSampler({SlotType::Sampler, entry.SamplerSlot}, *entry.Image, entry.Sampler.get());
    }
}

void ReleaseCubeSamplerSlot(mtl::BindlessSet &slots, uint32_t sampler_slot) {
    if (sampler_slot == InvalidSlot) return;
    slots.Release({SlotType::CubeSampler, sampler_slot});
}

void ReleaseEnvironmentSamplerSlots(mtl::BindlessSet &slots, const EnvironmentStore &environments) {
    for (const auto &hdri : environments.Hdris) {
        if (hdri.Prefiltered) {
            ReleaseCubeSamplerSlot(slots, hdri.Prefiltered->DiffuseEnv.SamplerSlot);
            ReleaseCubeSamplerSlot(slots, hdri.Prefiltered->SpecularEnv.SamplerSlot);
        }
    }
    if (environments.ImportedSceneWorld) {
        ReleaseCubeSamplerSlot(slots, environments.ImportedSceneWorld->DiffuseEnv.SamplerSlot);
        ReleaseCubeSamplerSlot(slots, environments.ImportedSceneWorld->SpecularEnv.SamplerSlot);
    }
    ReleaseCubeSamplerSlot(slots, environments.EmptySceneWorld.DiffuseEnv.SamplerSlot);
    ReleaseCubeSamplerSlot(slots, environments.EmptySceneWorld.SpecularEnv.SamplerSlot);
    for (const auto *tex : {&environments.BrdfLut, &environments.SheenELut, &environments.CharlieLut}) {
        if (tex->SamplerSlot != InvalidSlot) slots.Release({SlotType::Sampler, tex->SamplerSlot});
    }
}

namespace {
TextureEntry CreateTextureEntryAtSlot(
    const mtl::Context &ctx,
    TextureUploadBatch &batch,
    mtl::BindlessSet &slots,
    uint32_t pre_allocated_slot,
    std::span<const std::byte> pixels_rgba8,
    uint32_t width, uint32_t height,
    std::string name,
    TextureColorSpace color_space,
    MTL::SamplerAddressMode wrap_s, MTL::SamplerAddressMode wrap_t,
    const SamplerConfig &sampler_cfg, float max_anisotropy
) {
    const auto texture_format = ToTextureFormat(color_space);
    const uint32_t mip_levels = sampler_cfg.UsesMipmaps ? mtl::MipLevelCount(width, height) : 1u;

    // Levels above 0 are rendered into, and level 0 is uploaded here, so the texture stays shared.
    const auto usage = mip_levels > 1 ? MTL::TextureUsageShaderRead | MTL::TextureUsageRenderTarget : MTL::TextureUsageShaderRead;
    auto image = mtl::CreateTexture2D(ctx, texture_format, {width, height}, usage, mip_levels, MTL::StorageModeShared);
    mtl::Upload(image, 0, pixels_rgba8, width * 4u);
    GenerateMipChain(batch, image, texture_format);

    auto sampler = MakeSampler(ctx, sampler_cfg, wrap_s, wrap_t, max_anisotropy);
    slots.SetSampler({SlotType::Sampler, pre_allocated_slot}, *image, sampler.get());

    return {.Image = std::move(image), .Sampler = std::move(sampler), .SamplerSlot = pre_allocated_slot, .Config = sampler_cfg, .WrapS = wrap_s, .WrapT = wrap_t, .Name = std::move(name)};
}
} // namespace

TextureEntry CreateTextureEntry(
    const mtl::Context &ctx,
    TextureUploadBatch &batch,
    mtl::BindlessSet &slots,
    std::span<const std::byte> pixels_rgba8,
    uint32_t width, uint32_t height,
    std::string name,
    TextureColorSpace color_space,
    MTL::SamplerAddressMode wrap_s, MTL::SamplerAddressMode wrap_t,
    const SamplerConfig &sampler_cfg, float max_anisotropy
) {
    return CreateTextureEntryAtSlot(ctx, batch, slots, slots.Allocate(SlotType::Sampler), pixels_rgba8, width, height, std::move(name), color_space, wrap_s, wrap_t, sampler_cfg, max_anisotropy);
}

std::expected<TextureEntry, std::string> CreateTextureEntryFromEncoded(
    const mtl::Context &ctx, TextureUploadBatch &batch, mtl::BindlessSet &slots,
    std::span<const std::byte> encoded_bytes, std::string_view encoded_name, std::string texture_name,
    TextureColorSpace color_space,
    MTL::SamplerAddressMode wrap_s, MTL::SamplerAddressMode wrap_t,
    const SamplerConfig &sampler_cfg, float max_anisotropy
) {
    auto decoded = DecodeImageRgba8(encoded_bytes, encoded_name);
    if (!decoded) return std::unexpected{std::move(decoded.error())};
    return CreateTextureEntry(ctx, batch, slots, decoded->Pixels, decoded->Width, decoded->Height, std::move(texture_name), color_space, wrap_s, wrap_t, sampler_cfg, max_anisotropy);
}

uint32_t AllocateSamplerSlot(mtl::BindlessSet &slots) { return slots.Allocate(SlotType::Sampler); }
std::pair<uint32_t, uint32_t> AllocateIblCubeSlots(mtl::BindlessSet &slots) {
    return {slots.Allocate(SlotType::CubeSampler), slots.Allocate(SlotType::CubeSampler)};
}

std::expected<EnvironmentPrefiltered, std::string> MaterializeEnvironmentImport(
    const mtl::Context &ctx, mtl::BindlessSet &slots,
    const PendingEnvironmentImport &pending, const std::vector<gltf::Image> &images
) {
    const auto &ibl = pending.Source;
    std::vector<CubemapMipFacesF32> specular_mips;
    specular_mips.reserve(ibl.SpecularImageIndicesByMip.size());
    uint32_t specular_base_size = 0u;
    for (uint32_t mip = 0; mip < ibl.SpecularImageIndicesByMip.size(); ++mip) {
        CubemapMipFacesF32 faces{};
        for (uint32_t face = 0; face < 6u; ++face) {
            const auto image_index = ibl.SpecularImageIndicesByMip[mip][face];
            if (image_index >= images.size()) return std::unexpected{std::format("EXT_lights_image_based '{}' references image index {} (out of range).", ibl.Name, image_index)};

            const auto &src_image = images[image_index];
            auto decoded = DecodeImageRgba32f(
                src_image.Bytes,
                src_image.Name.empty() ? std::format("Image{}", image_index) : src_image.Name
            );
            if (!decoded) return std::unexpected{std::format("Failed to decode EXT_lights_image_based '{}' image {}: {}", ibl.Name, image_index, decoded.error())};
            if (decoded->Width != decoded->Height) return std::unexpected{std::format("EXT_lights_image_based '{}' image {} must be square (got {}x{}).", ibl.Name, image_index, decoded->Width, decoded->Height)};
            faces[face] = std::move(*decoded);
        }
        // Normalize EXT_lights_image_based face data to our cubemap upload convention.
        for (auto &face : faces) {
            if (face.Width == 0u || face.Height < 2u) continue;
            const size_t row_float_count = size_t(face.Width) * 4u;
            for (uint32_t y = 0; y < face.Height / 2u; ++y) {
                auto *row0 = face.Pixels.data() + size_t(y) * row_float_count;
                auto *row1 = face.Pixels.data() + size_t(face.Height - 1u - y) * row_float_count;
                std::swap_ranges(row0, row0 + row_float_count, row1);
            }
        }

        if (mip == 0u) {
            specular_base_size = faces[0].Width;
            if (ibl.SpecularImageSize != 0u && faces[0].Width != ibl.SpecularImageSize) {
                return std::unexpected{std::format(
                    "EXT_lights_image_based '{}' specularImageSize is {} but mip 0 face is {}x{}.",
                    ibl.Name, ibl.SpecularImageSize, faces[0].Width, faces[0].Width
                )};
            }
        }
        const uint32_t expected_size = std::max(1u, specular_base_size >> mip);
        if (faces[0].Width != expected_size) {
            return std::unexpected{std::format("EXT_lights_image_based '{}' mip {} has size {} but expected {}.", ibl.Name, mip, faces[0].Width, expected_size)};
        }
        specular_mips.emplace_back(std::move(faces));
    }

    auto specular_env = CreateCubemapEntryFromMipFacesF32(ctx, slots, pending.SpecularCubeSlot, specular_mips, ibl.Name + "_specular");
    if (!specular_env) return std::unexpected{std::move(specular_env.error())};

    std::vector<CubemapMipFacesF32> diffuse_mips;
    diffuse_mips.reserve(1);
    if (ibl.IrradianceCoefficients) diffuse_mips.emplace_back(BuildDiffuseCubemapFromIrradiance(*ibl.IrradianceCoefficients));
    else diffuse_mips.emplace_back(specular_mips.back());

    auto diffuse_env = CreateCubemapEntryFromMipFacesF32(ctx, slots, pending.DiffuseCubeSlot, diffuse_mips, ibl.Name + "_diffuse");
    if (!diffuse_env) return std::unexpected{std::move(diffuse_env.error())};

    return EnvironmentPrefiltered{.DiffuseEnv = std::move(*diffuse_env), .SpecularEnv = std::move(*specular_env), .Name = ibl.Name};
}

EnvironmentPrefiltered BuildFlatColorEnvironment(
    const mtl::Context &ctx, mtl::BindlessSet &slots,
    vec3 color, std::string name
) {
    CubemapMipFacesF32 face{};
    for (uint32_t f = 0; f < 6u; ++f) {
        face[f].Width = 1;
        face[f].Height = 1;
        face[f].Pixels = {color.x, color.y, color.z, 1.f};
    }
    const std::vector<CubemapMipFacesF32> mips{face};
    const auto [diffuse_slot, specular_slot] = AllocateIblCubeSlots(slots);
    auto specular = CreateCubemapEntryFromMipFacesF32(ctx, slots, specular_slot, mips, name + "_specular");
    auto diffuse = CreateCubemapEntryFromMipFacesF32(ctx, slots, diffuse_slot, mips, name + "_diffuse");
    if (!specular || !diffuse) throw std::runtime_error(std::format("Failed to build flat-color environment '{}'", name));
    return EnvironmentPrefiltered{.DiffuseEnv = std::move(*diffuse), .SpecularEnv = std::move(*specular), .Name = std::move(name)};
}

// Build diffuse and GGX-specular cubemaps from an equirectangular environment.
EnvironmentPrefiltered CreateIblFromHdri(
    const mtl::Context &ctx, mtl::BindlessSet &slots,
    const IblPrefilterPipelines &prefilter,
    const std::filesystem::path &path, std::string name
) {
    const auto path_str = path.string();
    auto decoded = DecodeImageFileRgba32f(path, path_str);
    if (!decoded) throw std::runtime_error(std::format("Failed to load HDR '{}': {}", path_str, decoded.error()));

    constexpr auto rgba32f = MTL::PixelFormatRGBA32Float;
    const uint32_t eq_w = decoded->Width, eq_h = decoded->Height;

    auto equirect = mtl::CreateTexture2D(ctx, rgba32f, {eq_w, eq_h}, MTL::TextureUsageShaderRead);
    mtl::Upload(equirect, 0, std::span<const std::byte>{reinterpret_cast<const std::byte *>(decoded->Pixels.data()), decoded->Pixels.size() * sizeof(float)}, eq_w * 4u * sizeof(float));

    const uint32_t raw_size = 512, raw_mips = mtl::MipLevelCount(raw_size, raw_size);
    auto raw_cube = mtl::CreateTextureCube(ctx, rgba32f, raw_size, MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite, raw_mips);
    auto raw_cube_write = mtl::CreateCubeMipView(raw_cube, 0);

    const uint32_t diff_size = 32;
    auto diff_cube = mtl::CreateTextureCube(ctx, rgba32f, diff_size, MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite);
    auto diff_write = mtl::CreateCubeMipView(diff_cube, 0);

    const uint32_t spec_size = 256, spec_mips = mtl::MipLevelCount(spec_size, spec_size);
    auto spec_cube = mtl::CreateTextureCube(ctx, rgba32f, spec_size, MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite, spec_mips);
    std::vector<NS::SharedPtr<MTL::Texture>> spec_writes;
    spec_writes.reserve(spec_mips);
    for (uint32_t mip = 0; mip < spec_mips; ++mip) {
        spec_writes.emplace_back(mtl::CreateCubeMipView(spec_cube, mip));
    }

    auto equirect_sampler = MakeLinearSampler(ctx, MTL::SamplerAddressModeRepeat);
    auto raw_cube_sampler = MakeLinearSampler(ctx, MTL::SamplerAddressModeClampToEdge);

    const auto prefilter_faces = [](
                                     MTL::ComputeCommandEncoder *compute, const mtl::ComputePipeline &pipeline,
                                     MTL::Texture *source, MTL::SamplerState *sampler, MTL::Texture *target, const auto &pc, uint32_t face_size
                                 ) {
        compute->setComputePipelineState(pipeline.State());
        compute->setTexture(source, 0);
        compute->setSamplerState(sampler, 0);
        compute->setTexture(target, 1);
        compute->setBytes(&pc, sizeof(pc), 0);
        compute->dispatchThreadgroups(MTL::Size((face_size + 7) / 8, (face_size + 7) / 8, 6), MTL::Size(8, 8, 1));
    };

    auto *command_buffer = ctx.Queue->commandBuffer();
    {
        auto *compute = command_buffer->computeCommandEncoder();
        prefilter_faces(compute, prefilter.EquirectToCubemap, *equirect, equirect_sampler.get(), raw_cube_write.get(), raw_size, raw_size);
        compute->endEncoding();
    }
    {
        auto *blit = command_buffer->blitCommandEncoder();
        blit->generateMipmaps(*raw_cube);
        blit->endEncoding();
    }
    {
        auto *compute = command_buffer->computeCommandEncoder();
        prefilter_faces(compute, prefilter.DiffuseIrradiance, *raw_cube, raw_cube_sampler.get(), diff_write.get(), diff_size, diff_size);

        for (uint32_t mip = 0; mip < spec_mips; ++mip) {
            const uint32_t mip_face_size = std::max(1u, spec_size >> mip);
            struct SpecPC {
                uint32_t FaceSize, SourceSize;
                float Roughness;
            };
            const SpecPC pc{.FaceSize = mip_face_size, .SourceSize = raw_size, .Roughness = float(mip) / float(spec_mips - 1)};
            prefilter_faces(compute, prefilter.SpecularPrefilter, *raw_cube, raw_cube_sampler.get(), spec_writes[mip].get(), pc, mip_face_size);
        }
        compute->endEncoding();
    }
    // The temporaries above go out of scope on return, so the work settles first.
    command_buffer->commit();
    command_buffer->waitUntilCompleted();

    auto diff_sampler = MakeLinearSampler(ctx, MTL::SamplerAddressModeClampToEdge);
    auto spec_sampler = MakeLinearSampler(ctx, MTL::SamplerAddressModeClampToEdge);
    const auto diff_slot = slots.Allocate(SlotType::CubeSampler);
    const auto spec_slot = slots.Allocate(SlotType::CubeSampler);
    slots.SetSampler({SlotType::CubeSampler, diff_slot}, *diff_cube, diff_sampler.get());
    slots.SetSampler({SlotType::CubeSampler, spec_slot}, *spec_cube, spec_sampler.get());
    return {
        .DiffuseEnv = {.Image = std::move(diff_cube), .Sampler = std::move(diff_sampler), .SamplerSlot = diff_slot, .Name = name + "_diffuse"},
        .SpecularEnv = {.Image = std::move(spec_cube), .Sampler = std::move(spec_sampler), .SamplerSlot = spec_slot, .Name = name + "_specular"},
        .Name = std::move(name),
    };
}

IblSamplers MakeIblSamplers(const EnvironmentPrefiltered &pre, const EnvironmentStore &environments) {
    return {
        .DiffuseEnvSamplerSlot = pre.DiffuseEnv.SamplerSlot,
        .SpecularEnvSamplerSlot = pre.SpecularEnv.SamplerSlot,
        .BrdfLutSamplerSlot = environments.BrdfLut.SamplerSlot,
        .SpecularEnvMipCount = pre.SpecularEnv.Image.MipLevels,
        .SheenEnvSamplerSlot = pre.SpecularEnv.SamplerSlot,
        .SheenEnvMipCount = pre.SpecularEnv.Image.MipLevels,
        .SheenELutSamplerSlot = environments.SheenELut.SamplerSlot,
        .CharlieLutSamplerSlot = environments.CharlieLut.SamplerSlot,
    };
}

std::vector<std::byte> ReadbackImageRgba8(const mtl::Context &ctx, const mtl::Texture &texture, uint32_t x, uint32_t y, mtl::Extent2D extent) {
    const size_t byte_size = size_t(extent.Width) * extent.Height * 4u;
    std::vector<std::byte> out(byte_size);
    // A private attachment cannot be read directly, so it blits into a shared texture first.
    if (texture.Handle->storageMode() == MTL::StorageModePrivate) {
        const auto staging = mtl::CreateTexture2D(ctx, texture.Handle->pixelFormat(), extent, MTL::TextureUsageShaderRead);
        mtl::CopyTextureRegion(ctx, texture, x, y, extent, staging);
        staging.Handle->getBytes(out.data(), extent.Width * 4u, MTL::Region::Make2D(0, 0, extent.Width, extent.Height), 0);
        return out;
    }
    texture.Handle->getBytes(out.data(), extent.Width * 4u, MTL::Region::Make2D(x, y, extent.Width, extent.Height), 0);
    return out;
}

std::expected<std::vector<std::byte>, std::string> ReadbackTextureRgba8(const mtl::Context &ctx, const TextureEntry &entry) {
    if (entry.Image.Extent.Width == 0 || entry.Image.Extent.Height == 0) {
        return std::unexpected{std::format("Texture '{}' has zero dimension {}x{}.", entry.Name, entry.Image.Extent.Width, entry.Image.Extent.Height)};
    }
    return ReadbackImageRgba8(ctx, entry.Image, 0, 0, entry.Image.Extent);
}

std::expected<TextureEntry, std::string> MaterializeTextureEntry(
    const mtl::Context &ctx,
    TextureUploadBatch &batch, mtl::BindlessSet &slots,
    const PendingTextureUpload &item, const std::vector<gltf::Image> &gltf_images, float max_anisotropy
) {
    if (const auto *raw = std::get_if<PendingTextureUpload::RawPixels>(&item.Source)) {
        return CreateTextureEntryAtSlot(
            ctx, batch, slots, item.SamplerSlot,
            raw->Pixels, raw->Width, raw->Height, item.Name,
            item.ColorSpace, item.WrapS, item.WrapT, item.Sampler, max_anisotropy
        );
    }
    const auto &ref = std::get<PendingTextureUpload::GltfImageRef>(item.Source);
    if (ref.ImageIndex >= gltf_images.size()) {
        return std::unexpected{std::format("PendingTextureUpload '{}' references gltf image index {} (out of range; {} images).", item.Name, ref.ImageIndex, gltf_images.size())};
    }
    const auto &source = gltf_images[ref.ImageIndex];
    if (source.MimeType != gltf::MimeType::KTX2) {
        auto decoded = DecodeImageRgba8(source.Bytes, source.Name);
        if (!decoded) return std::unexpected{std::move(decoded.error())};
        auto entry = CreateTextureEntryAtSlot(
            ctx, batch, slots, item.SamplerSlot,
            decoded->Pixels, decoded->Width, decoded->Height, item.Name,
            item.ColorSpace, item.WrapS, item.WrapT, item.Sampler, max_anisotropy
        );
        entry.SourceImageIndex = ref.ImageIndex;
        return entry;
    }

    basist::basisu_transcoder_init();

    basist::ktx2_transcoder transcoder;
    if (!transcoder.init(source.Bytes.data(), uint32_t(source.Bytes.size()))) return std::unexpected{std::format("Failed to parse KTX2 image '{}'.", source.Name)};
    if (!transcoder.start_transcoding()) return std::unexpected{std::format("Failed to start transcoding KTX2 image '{}'.", source.Name)};

    const auto [texture_format, basis_fmt] = SelectKtx2Format(ctx, item.ColorSpace);
    const uint32_t width = transcoder.get_width(), height = transcoder.get_height();
    const uint32_t mip_levels = transcoder.get_levels();

    std::vector<std::byte> all_mip_data;
    std::vector<MipUpload> mips;
    mips.reserve(mip_levels);
    size_t offset = 0;
    const uint32_t block_bytes = basist::basis_get_bytes_per_block_or_pixel(basis_fmt);
    // A block format addresses rows of blocks, an uncompressed one rows of pixels.
    const bool block_compressed = !basist::basis_transcoder_format_is_uncompressed(basis_fmt);
    for (uint32_t mip = 0; mip < mip_levels; ++mip) {
        const uint32_t mip_w = std::max(1u, width >> mip), mip_h = std::max(1u, height >> mip);
        const uint32_t mip_bytes = basist::basis_compute_transcoded_image_size_in_bytes(basis_fmt, mip_w, mip_h);
        const uint32_t block_count = mip_bytes / basist::basis_get_bytes_per_block_or_pixel(basis_fmt);

        const size_t prev_size = all_mip_data.size();
        all_mip_data.resize(prev_size + mip_bytes);
        if (!transcoder.transcode_image_level(mip, 0, 0, all_mip_data.data() + prev_size, block_count, basis_fmt)) {
            return std::unexpected{std::format("Failed to transcode KTX2 image '{}' mip {}.", source.Name, mip)};
        }

        const uint32_t bytes_per_row = block_compressed ? ((mip_w + 3u) / 4u) * block_bytes : mip_w * block_bytes;
        mips.emplace_back(mip, offset, mip_bytes, bytes_per_row);
        offset += mip_bytes;
    }

    auto entry = CreateCompressedTextureEntry(ctx, slots, item.SamplerSlot, all_mip_data, mips, texture_format, width, height, mip_levels, item.Name, item.WrapS, item.WrapT, item.Sampler, max_anisotropy);
    entry.SourceImageIndex = ref.ImageIndex;
    return entry;
}

TextureEntry CreateDefaultLutTexture(const mtl::Context &ctx, TextureUploadBatch &batch, mtl::BindlessSet &slots, const std::filesystem::path &lut_path, std::string_view name, float max_anisotropy) {
    const auto encoded = File::ReadAsString(lut_path).value_or(std::string{});
    const auto lut_path_str = lut_path.string();
    auto texture = CreateTextureEntryFromEncoded(
        ctx, batch, slots,
        std::as_bytes(std::span{encoded}), lut_path_str, std::string{name},
        TextureColorSpace::Linear, MTL::SamplerAddressModeClampToEdge, MTL::SamplerAddressModeClampToEdge,
        {.MinFilter = MTL::SamplerMinMagFilterLinear, .MagFilter = MTL::SamplerMinMagFilterLinear, .MipmapMode = MTL::SamplerMipFilterLinear, .UsesMipmaps = false}, max_anisotropy
    );
    if (!texture) throw std::runtime_error(std::format("Failed to initialize default LUT texture '{}': {}", lut_path_str, texture.error()));
    return std::move(*texture);
}

std::vector<TextureRef> GetTextureRefs(entt::registry &r) {
    const auto &store = r.ctx().get<TextureStore>();
    std::vector<TextureRef> refs;
    refs.reserve(store.Textures.size());
    for (const auto &t : store.Textures) refs.emplace_back(t.SamplerSlot, t.Name);
    return refs;
}

HdriRefs GetHdriRefs(entt::registry &r) {
    const auto &environments = r.ctx().get<EnvironmentStore>();
    HdriRefs refs;
    refs.ActiveIndex = environments.ActiveHdriIndex;
    refs.Names.reserve(environments.Hdris.size());
    for (const auto &hdri : environments.Hdris) refs.Names.emplace_back(hdri.Name);
    return refs;
}

void ImportObjPlyMaterials(entt::registry &r, std::span<const ObjPlyMaterial> materials, const std::filesystem::path &mesh_path, uint32_t mesh_store_id) {
    const auto &ctx = r.ctx().get<const mtl::Context>();
    auto &slots = r.ctx().get<mtl::BindlessSet>();
    auto &buffers = r.ctx().get<GpuBuffers>();
    auto &meshes = r.ctx().get<MeshStore>();
    auto &textures = r.ctx().get<TextureStore>();

    auto obj_batch = BeginTextureUploadBatch(ctx, r.ctx().get<mtl::LibraryCache>());
    std::unordered_map<std::string, uint32_t> texture_slot_cache;
    const auto resolve_texture_slot =
        [&](
            const std::optional<std::filesystem::path> &source_texture_path,
            TextureColorSpace color_space,
            std::string_view material_name, std::string_view texture_label
        ) -> uint32_t {
        if (!source_texture_path) return InvalidSlot;
        auto texture_path = *source_texture_path;
        if (texture_path.is_relative()) texture_path = mesh_path.parent_path() / texture_path;
        texture_path = texture_path.lexically_normal();

        const auto cache_key = std::format("{}|{}", texture_path.generic_string(), color_space == TextureColorSpace::Srgb ? "sRGB" : "Linear");
        if (const auto it = texture_slot_cache.find(cache_key); it != texture_slot_cache.end()) return it->second;

        const auto read = File::ReadAsString(texture_path);
        if (!read) {
            std::cerr << std::format(
                "Warning: Failed to read OBJ texture '{}' for material '{}' ({}) in '{}': {}\n",
                texture_path.string(), material_name, texture_label, mesh_path.string(), read.error()
            );
            return InvalidSlot;
        }
        const std::string &encoded = *read;

        auto texture = CreateTextureEntryFromEncoded(
            ctx,
            obj_batch,
            slots,
            std::as_bytes(std::span{encoded}),
            texture_path.filename().string(),
            std::format("{} ({})", texture_path.filename().string(), color_space == TextureColorSpace::Srgb ? "sRGB" : "Linear"),
            color_space,
            MTL::SamplerAddressModeRepeat,
            MTL::SamplerAddressModeRepeat,
            SamplerConfig{}, r.ctx().get<const ActiveSamplerAnisotropy>().Value
        );
        if (!texture) {
            std::cerr << std::format(
                "Warning: Failed to decode OBJ texture '{}' for material '{}' ({}) in '{}': {}\n",
                texture_path.string(), material_name, texture_label, mesh_path.string(), texture.error()
            );
            return InvalidSlot;
        }

        const auto sampler_slot = texture->SamplerSlot;
        textures.Textures.emplace_back(std::move(*texture));
        texture_slot_cache.emplace(cache_key, sampler_slot);
        return sampler_slot;
    };

    std::vector<uint32_t> scene_material_indices(materials.size(), 0u);
    std::vector<std::string> names;
    names.reserve(materials.size());
    buffers.Materials.ReserveElements(buffers.Materials.Count() + materials.size());
    for (uint32_t material_index = 0; material_index < materials.size(); ++material_index) {
        const auto &source = materials[material_index];
        const auto material_name = source.Name.empty() ? std::format("Material{}", material_index) : source.Name;
        const auto base_color_texture = resolve_texture_slot(source.BaseColorTexturePath, TextureColorSpace::Srgb, material_name, "baseColor");
        const auto normal_texture = resolve_texture_slot(source.NormalTexturePath, TextureColorSpace::Linear, material_name, "normal");
        scene_material_indices[material_index] = buffers.Materials.Append({
            .BaseColorFactor = source.BaseColorFactor,
            .MetallicFactor = std::clamp(source.MetallicFactor, 0.f, 1.f),
            .RoughnessFactor = std::clamp(source.RoughnessFactor, 0.f, 1.f),
            .AlphaMode = (source.BaseColorFactor.w < 1.f || source.HasAlphaTexture) ?
                MaterialAlphaMode::Blend :
                MaterialAlphaMode::Opaque,
            .BaseColorTexture = {.Slot = base_color_texture != InvalidSlot ? base_color_texture : textures.WhiteTextureSlot},
            .NormalTexture = {.Slot = normal_texture},
        });
        names.emplace_back(material_name);
    }
    SubmitTextureUploadBatch(obj_batch);

    auto &material_store = r.ctx().get<MaterialStore>();
    material_store.Names.insert(material_store.Names.end(), std::make_move_iterator(names.begin()), std::make_move_iterator(names.end()));

    if (auto primitive_materials = meshes.GetPrimitiveMaterialIndices(mesh_store_id); !primitive_materials.empty()) {
        const auto fallback = scene_material_indices.front();
        for (auto &primitive_material : primitive_materials) {
            primitive_material = primitive_material < scene_material_indices.size() ? scene_material_indices[primitive_material] : fallback;
        }
    }
}

void ResetImportedTexturesAndMaterials(entt::registry &r) {
    auto &slots = r.ctx().get<mtl::BindlessSet>();
    auto &buffers = r.ctx().get<GpuBuffers>();
    auto &textures = r.ctx().get<TextureStore>();
    // Index 0 is the default white texture (permanent); imported textures start at index 1.
    if (textures.Textures.size() > 1) {
        ReleaseSamplerSlots(slots, CollectSamplerSlots(std::span<const TextureEntry>{textures.Textures}.subspan(1)));
        textures.Textures.erase(textures.Textures.begin() + 1, textures.Textures.end());
    }
    textures.WhiteTextureSlot = textures.Textures.empty() ? InvalidSlot : textures.Textures.front().SamplerSlot;

    if (buffers.Materials.Count() > 1) buffers.Materials.SetCount(1u);
    if (auto &ms = r.ctx().get<MaterialStore>(); ms.Names.size() > 1) ms.Names.erase(ms.Names.begin() + 1, ms.Names.end());
}
