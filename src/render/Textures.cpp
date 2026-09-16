#include "render/Textures.h"

#include "File.h"
#include "gltf/Image.h"
#include "gpu/CubeFacePushConstants.h"
#include "gpu/PrefilterPushConstants.h"
#include "image/ImageDecode.h"
#include "mesh/MeshStore.h"
#include "metal/Bindless.h"
#include "metal/MetalCpp.h"
#include "project/Assets.h"
#include "render/GpuBuffers.h"
#include "render/MaterialComponents.h"
#include "render/Pipelines.h"
#include "render/TextureRefs.h"

#include "state/Scene.h"
#include <basisu_transcoder.h>

#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

namespace {
NS::SharedPtr<MTL::SamplerState> MakeLinearSampler(const mtl::Context &ctx, MTL::SamplerAddressMode address_mode) {
    return mtl::CreateSampler(ctx, MTL::SamplerMinMagFilterLinear, MTL::SamplerMipFilterLinear, address_mode);
}

NS::SharedPtr<MTL::SamplerState> MakeSampler(const mtl::Context &ctx, const TextureParams &params, float max_anisotropy) {
    // Anisotropic filtering only applies with a mip chain.
    const bool anisotropic = params.Sampler.UsesMipmaps && max_anisotropy > 1.f;
    return mtl::CreateSampler(ctx, {
                                       params.Sampler.MinFilter,
                                       params.Sampler.MagFilter,
                                       params.Sampler.MipmapMode,
                                       params.WrapS,
                                       params.WrapT,
                                       MTL::SamplerAddressModeRepeat,
                                       anisotropic ? max_anisotropy : 1.f,
                                   });
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

} // namespace

TextureUploadBatch BeginTextureUploadBatch(const mtl::Context &ctx) { return {.Cb = ctx.Queue->commandBuffer()}; }

void SubmitTextureUploadBatch(TextureUploadBatch &batch) {
    if (!batch.Cb) return;
    batch.Cb->commit();
    // Complete uploads before immediate readback and binding.
    batch.Cb->waitUntilCompleted();
    batch.Cb = nullptr;
}

void ReleaseTextureSlots(mtl::BindlessSet &slots, std::span<const TextureEntry> textures) {
    for (const auto &texture : textures) {
        if (texture.SamplerSlot != InvalidSlot) slots.Release({SlotType::Sampler, texture.SamplerSlot});
    }
}

float ClampMaxAnisotropy(float requested) { return std::clamp(requested, 1.f, MaxSamplerAnisotropy); }

void RebuildTextureSamplers(const mtl::Context &ctx, mtl::BindlessSet &slots, TextureStore &textures, float max_anisotropy) {
    for (auto &entry : textures.Textures) {
        entry.Sampler = MakeSampler(ctx, entry.Params, max_anisotropy);
        slots.SetSampler({SlotType::Sampler, entry.SamplerSlot}, *entry.Image, entry.Sampler.get());
    }
}

void ReleaseCubeSamplerSlot(mtl::BindlessSet &slots, uint32_t sampler_slot) {
    if (sampler_slot == InvalidSlot) return;
    slots.Release({SlotType::CubeSampler, sampler_slot});
}

void ResetImportedEnvironment(state::Scene &r) {
    auto &env = r.ctx().get<EnvironmentStore>();
    if (env.ImportedSceneWorld) {
        auto &slots = r.ctx().get<mtl::BindlessSet>();
        ReleaseCubeSamplerSlot(slots, env.ImportedSceneWorld->DiffuseEnv.SamplerSlot);
        ReleaseCubeSamplerSlot(slots, env.ImportedSceneWorld->SpecularEnv.SamplerSlot);
        env.ImportedSceneWorld.reset();
    }
    env.SceneWorldRotation = mat3{1.f};
    env.SceneWorld = {.Ibl = MakeIblSamplers(env.EmptySceneWorld, env), .Name = env.EmptySceneWorld.Name};
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
}

TextureEntry CreateTextureEntry(
    const mtl::Context &ctx, TextureUploadBatch &batch, mtl::BindlessSet &slots, uint32_t sampler_slot,
    const TexturePixels &pixels, TextureParams params, float max_anisotropy
) {
    mtl::Texture image;
    if (const auto *rgba = std::get_if<Rgba8Pixels>(&pixels)) {
        const uint32_t mip_levels = params.Sampler.UsesMipmaps ? mtl::MipLevelCount(rgba->Width, rgba->Height) : 1u;
        image = mtl::CreateTexture2D(ctx, ToTextureFormat(params.ColorSpace), {rgba->Width, rgba->Height}, MTL::TextureUsageShaderRead, mip_levels, MTL::StorageModeShared);
        mtl::Upload(image, 0, rgba->Pixels, rgba->Width * 4u);
        if (mip_levels > 1) {
            auto *blit = batch.Cb->blitCommandEncoder();
            blit->generateMipmaps(*image);
            blit->endEncoding();
        }
    } else {
        const auto &ktx = std::get<Ktx2Pixels>(pixels);
        image = mtl::CreateTexture2D(ctx, ktx.Format, {ktx.Width, ktx.Height}, MTL::TextureUsageShaderRead, uint32_t(ktx.Mips.size()));
        for (const auto &mip : ktx.Mips) mtl::Upload(image, mip.Level, ktx.Data.subspan(mip.Offset, mip.Bytes), mip.BytesPerRow);
    }
    auto sampler = MakeSampler(ctx, params, max_anisotropy);
    slots.SetSampler({SlotType::Sampler, sampler_slot}, *image, sampler.get());
    return {.Image = std::move(image), .Sampler = std::move(sampler), .SamplerSlot = sampler_slot, .Params = std::move(params)};
}

uint32_t AllocateSamplerSlot(mtl::BindlessSet &slots) { return slots.Allocate(SlotType::Sampler); }
std::pair<uint32_t, uint32_t> AllocateIblCubeSlots(mtl::BindlessSet &slots) {
    return {slots.Allocate(SlotType::CubeSampler), slots.Allocate(SlotType::CubeSampler)};
}

std::expected<EnvironmentPrefiltered, std::string> MaterializeEnvironmentImport(
    const state::Scene &r, mtl::BindlessSet &slots,
    const PendingEnvironmentImport &pending, const std::vector<gltf::Image> &images
) {
    const auto &ctx = r.ctx().get<const mtl::Context>();
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
            std::vector<std::byte> loaded;
            std::span<const std::byte> bytes = src_image.Bytes;
            if (bytes.empty()) {
                auto file = File::Read(project::ResolveAsset(r, src_image.SourcePath));
                if (!file) return std::unexpected{file.error()};
                loaded = std::move(*file);
                bytes = loaded;
            }
            auto decoded = DecodeImageRgba32f(
                bytes,
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
    const mtl::Context &ctx, mtl::BindlessSet &slots, const Pipelines &pipelines, const std::filesystem::path &path, std::string name
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
    std::vector<mtl::Texture> spec_writes;
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
        prefilter_faces(compute, pipelines.EquirectToCubemap, *equirect, equirect_sampler.get(), *raw_cube_write, CubeFacePushConstants{.FaceSize = raw_size}, raw_size);
        compute->endEncoding();
    }
    {
        auto *blit = command_buffer->blitCommandEncoder();
        blit->generateMipmaps(*raw_cube);
        blit->endEncoding();
    }
    {
        auto *compute = command_buffer->computeCommandEncoder();
        prefilter_faces(compute, pipelines.DiffuseIrradiance, *raw_cube, raw_cube_sampler.get(), *diff_write, CubeFacePushConstants{.FaceSize = diff_size}, diff_size);

        for (uint32_t mip = 0; mip < spec_mips; ++mip) {
            const uint32_t mip_face_size = std::max(1u, spec_size >> mip);
            const PrefilterPushConstants pc{.FaceSize = mip_face_size, .SourceSize = raw_size, .Roughness = float(mip) / float(spec_mips - 1)};
            prefilter_faces(compute, pipelines.SpecularPrefilter, *raw_cube, raw_cube_sampler.get(), *spec_writes[mip], pc, mip_face_size);
        }
        compute->endEncoding();
    }
    // Complete GPU work before releasing temporary textures.
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

void SetStudioEnvironment(state::Scene &r, uint32_t index) {
    const auto &ctx = r.ctx().get<const mtl::Context>();
    auto &slots = r.ctx().get<mtl::BindlessSet>();
    auto &environments = r.ctx().get<EnvironmentStore>();
    auto &hdri = environments.Hdris[index];
    if (!hdri.Prefiltered) hdri.Prefiltered = CreateIblFromHdri(ctx, slots, GetPipelines(r), hdri.Path, hdri.Name);
    const auto &pre = *hdri.Prefiltered;
    environments.ActiveHdriIndex = index;
    environments.StudioWorld = {.Ibl = MakeIblSamplers(pre, environments), .Name = hdri.Name};
}

void SetStudioEnvironment(state::Scene &r, std::string_view name) {
    const auto &hdris = r.ctx().get<const EnvironmentStore>().Hdris;
    const auto it = std::ranges::find(hdris, name, &HdriEntry::Name);
    SetStudioEnvironment(r, it != hdris.end() ? uint32_t(std::distance(hdris.begin(), it)) : 0u);
}

void RebuildStudioEnvironments(state::Scene &r) {
    auto &slots = r.ctx().get<mtl::BindlessSet>();
    auto &environments = r.ctx().get<EnvironmentStore>();
    if (environments.Hdris.empty()) return;
    for (auto &hdri : environments.Hdris) {
        if (!hdri.Prefiltered) continue;
        ReleaseCubeSamplerSlot(slots, hdri.Prefiltered->DiffuseEnv.SamplerSlot);
        ReleaseCubeSamplerSlot(slots, hdri.Prefiltered->SpecularEnv.SamplerSlot);
        hdri.Prefiltered.reset();
    }
    SetStudioEnvironment(r, environments.ActiveHdriIndex);
}

IblSamplers MakeIblSamplers(const EnvironmentPrefiltered &pre, const EnvironmentStore &environments) {
    return {
        .DiffuseEnvSamplerSlot = pre.DiffuseEnv.SamplerSlot,
        .SpecularEnvSamplerSlot = pre.SpecularEnv.SamplerSlot,
        .BrdfLutSamplerSlot = environments.BrdfLutSlot,
        .SpecularEnvMipCount = pre.SpecularEnv.Image.MipLevels,
        .SheenEnvSamplerSlot = pre.SpecularEnv.SamplerSlot,
        .SheenEnvMipCount = pre.SpecularEnv.Image.MipLevels,
        .SheenELutSamplerSlot = environments.SheenELutSlot,
        .CharlieLutSamplerSlot = environments.CharlieLutSlot,
    };
}

std::vector<std::byte> ReadbackImageRgba8(const mtl::Context &ctx, const mtl::Texture &texture, uint32_t x, uint32_t y, mtl::Extent2D extent) {
    const size_t byte_size = size_t(extent.Width) * extent.Height * 4u;
    std::vector<std::byte> out(byte_size);
    // A private attachment cannot be read directly, so it blits into a shared texture first.
    if (texture.Handle->storageMode() == MTL::StorageModePrivate) {
        const auto staging = mtl::CreateUntrackedTexture2D(ctx, texture.Handle->pixelFormat(), extent, MTL::TextureUsageShaderRead);
        if (!mtl::CopyTextureRegion(ctx, texture, x, y, extent, staging)) {
            throw std::runtime_error("Failed to read back a private Metal texture.");
        }
        staging.Handle->getBytes(out.data(), extent.Width * 4u, MTL::Region::Make2D(0, 0, extent.Width, extent.Height), 0);
        return out;
    }
    texture.Handle->getBytes(out.data(), extent.Width * 4u, MTL::Region::Make2D(x, y, extent.Width, extent.Height), 0);
    return out;
}

std::expected<std::vector<std::byte>, std::string> ReadbackTextureRgba8(const mtl::Context &ctx, const TextureEntry &entry) {
    if (entry.Image.Extent.Width == 0 || entry.Image.Extent.Height == 0) {
        return std::unexpected{std::format("Texture '{}' has zero dimension {}x{}.", entry.Params.Name, entry.Image.Extent.Width, entry.Image.Extent.Height)};
    }
    return ReadbackImageRgba8(ctx, entry.Image, 0, 0, entry.Image.Extent);
}

std::expected<TextureEntry, std::string> MaterializeTextureEntry(
    const state::Scene &r,
    TextureUploadBatch &batch, mtl::BindlessSet &slots,
    const PendingTextureUpload &item, const std::vector<gltf::Image> &gltf_images, float max_anisotropy
) {
    const auto &ctx = r.ctx().get<const mtl::Context>();
    if (const auto *raw = std::get_if<PendingTextureUpload::RawPixels>(&item.Source)) {
        return CreateTextureEntry(ctx, batch, slots, item.SamplerSlot, Rgba8Pixels{raw->Pixels, raw->Width, raw->Height}, item.Params, max_anisotropy);
    }
    const auto &ref = std::get<PendingTextureUpload::GltfImageRef>(item.Source);
    if (ref.ImageIndex >= gltf_images.size()) {
        return std::unexpected{std::format("PendingTextureUpload '{}' references gltf image index {} (out of range; {} images).", item.Params.Name, ref.ImageIndex, gltf_images.size())};
    }
    const auto &source = gltf_images[ref.ImageIndex];
    std::vector<std::byte> loaded;
    std::span<const std::byte> bytes = source.Bytes;
    if (bytes.empty()) {
        auto file = File::Read(project::ResolveAsset(r, source.SourcePath));
        if (!file) return std::unexpected{file.error()};
        loaded = std::move(*file);
        bytes = loaded;
    }
    if (source.MimeType != gltf::MimeType::KTX2) {
        auto decoded = DecodeImageRgba8(bytes, source.Name);
        if (!decoded) return std::unexpected{std::move(decoded.error())};
        auto entry = CreateTextureEntry(ctx, batch, slots, item.SamplerSlot, Rgba8Pixels{decoded->Pixels, decoded->Width, decoded->Height}, item.Params, max_anisotropy);
        entry.SourceImageIndex = ref.ImageIndex;
        return entry;
    }

    basist::basisu_transcoder_init();

    basist::ktx2_transcoder transcoder;
    if (!transcoder.init(bytes.data(), uint32_t(bytes.size()))) return std::unexpected{std::format("Failed to parse KTX2 image '{}'.", source.Name)};
    if (!transcoder.start_transcoding()) return std::unexpected{std::format("Failed to start transcoding KTX2 image '{}'.", source.Name)};

    const auto [texture_format, basis_fmt] = SelectKtx2Format(ctx, item.Params.ColorSpace);
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

    auto entry = CreateTextureEntry(ctx, batch, slots, item.SamplerSlot, Ktx2Pixels{texture_format, width, height, all_mip_data, mips}, item.Params, max_anisotropy);
    entry.SourceImageIndex = ref.ImageIndex;
    return entry;
}

uint32_t QueueLutTexture(TextureStore &textures, mtl::BindlessSet &slots, const std::filesystem::path &lut_path, std::string name) {
    const auto lut_path_str = lut_path.string();
    const auto encoded = File::Read(lut_path);
    if (!encoded) throw std::runtime_error(std::format("Failed to read default LUT texture '{}': {}", lut_path_str, encoded.error()));
    auto decoded = DecodeImageRgba8(*encoded, lut_path_str);
    if (!decoded) throw std::runtime_error(std::format("Failed to decode default LUT texture '{}': {}", lut_path_str, decoded.error()));
    const auto slot = AllocateSamplerSlot(slots);
    textures.PendingUploads.emplace_back(PendingTextureUpload{
        .SamplerSlot = slot,
        .Source = PendingTextureUpload::RawPixels{std::move(decoded->Pixels), decoded->Width, decoded->Height},
        .Params = {
            .ColorSpace = TextureColorSpace::Linear,
            .WrapS = MTL::SamplerAddressModeClampToEdge,
            .WrapT = MTL::SamplerAddressModeClampToEdge,
            .Sampler = {.MinFilter = MTL::SamplerMinMagFilterLinear, .MagFilter = MTL::SamplerMinMagFilterLinear, .MipmapMode = MTL::SamplerMipFilterLinear, .UsesMipmaps = false},
            .Name = std::move(name),
        },
    });
    return slot;
}

std::vector<TextureRef> GetTextureRefs(state::Scene &r) {
    const auto &store = r.ctx().get<TextureStore>();
    std::vector<TextureRef> refs;
    refs.reserve(store.Textures.size());
    for (const auto &t : store.Textures) refs.emplace_back(t.SamplerSlot, t.Params.Name);
    return refs;
}

HdriRefs GetHdriRefs(state::Scene &r) {
    const auto &environments = r.ctx().get<EnvironmentStore>();
    HdriRefs refs;
    refs.ActiveIndex = environments.ActiveHdriIndex;
    refs.Names.reserve(environments.Hdris.size());
    for (const auto &hdri : environments.Hdris) refs.Names.emplace_back(hdri.Name);
    return refs;
}

void ReleaseImportedTextures(state::Scene &r) {
    auto &slots = r.ctx().get<mtl::BindlessSet>();
    auto &textures = r.ctx().get<TextureStore>();
    // The raw-pixel entries materialized at engine init lead the list, and every imported entry follows them.
    const auto imported = std::ranges::find_if(textures.Textures, [](const auto &t) { return t.SourceImageIndex != UINT32_MAX; });
    ReleaseTextureSlots(slots, std::span<const TextureEntry>{imported, textures.Textures.end()});
    textures.Textures.erase(imported, textures.Textures.end());
    textures.WhiteTextureSlot = textures.Textures.empty() ? InvalidSlot : textures.Textures.front().SamplerSlot;
}

void ResetImportedTexturesAndMaterials(state::Scene &r) {
    ReleaseImportedTextures(r);
    auto &buffers = r.ctx().get<GpuBuffers>();
    if (buffers.Materials.Count<PBRMaterial>() > 1) buffers.Materials.SetCount<PBRMaterial>(1u);
    if (auto &ms = r.ctx().get<MaterialStore>(); ms.Names.size() > 1) ms.ResizeNames(1);
}
