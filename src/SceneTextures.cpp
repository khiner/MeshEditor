#include "SceneTextures.h"
#include "Bindless.h"
#include "File.h"
#include "ImageDecode.h"
#include "scene_impl/IblPrefilterPipelines.h"
#include "vulkan/Buffer.h"

#include <glm/geometric.hpp>

#include <array>
#include <bit>
#include <format>
#include <stdexcept>

namespace {
void Submit(vk::Queue queue, vk::CommandBuffer command_buffer, vk::Fence fence, vk::Device device) {
    vk::SubmitInfo submit{};
    submit.setCommandBuffers(command_buffer);
    queue.submit(submit, fence);
    if (auto wait_result = device.waitForFences(fence, VK_TRUE, UINT64_MAX); wait_result != vk::Result::eSuccess) {
        throw std::runtime_error(std::format("Failed to wait for fence: {}", vk::to_string(wait_result)));
    }
    device.resetFences(fence);
}

vk::Format ToTextureFormat(TextureColorSpace color_space) { return color_space == TextureColorSpace::Srgb ? vk::Format::eR8G8B8A8Srgb : vk::Format::eR8G8B8A8Unorm; }

vec3 CubemapFaceDirection(uint32_t face, float u, float v) {
    switch (face) {
        case 0: return glm::normalize(vec3{1.f, -v, -u}); // +X
        case 1: return glm::normalize(vec3{-1.f, -v, u}); // -X
        case 2: return glm::normalize(vec3{u, 1.f, v}); // +Y
        case 3: return glm::normalize(vec3{u, -1.f, -v}); // -Y
        case 4: return glm::normalize(vec3{u, -v, 1.f}); // +Z
        default: return glm::normalize(vec3{-u, -v, -1.f}); // -Z
    }
}

// EXT_lights_image_based Appendix B (Romain Guy) irradiance reconstruction constants.
vec3 EvaluateIrradianceSH(const std::array<vec3, 9> &l, vec3 n) {
    static constexpr float c0 = 0.886227f;
    static constexpr float c1 = 1.023327f;
    static constexpr float c2 = 0.858086f;
    static constexpr float c3 = 0.247708f;
    static constexpr float c4 = 0.429043f;
    const float x = n.x, y = n.y, z = n.z;
    const vec3 irradiance =
        c0 * l[0] -
        c1 * y * l[1] +
        c1 * z * l[2] -
        c1 * x * l[3] +
        c2 * x * y * l[4] -
        c2 * y * z * l[5] +
        c3 * (3.f * z * z - 1.f) * l[6] -
        c2 * x * z * l[7] +
        c4 * (x * x - y * y) * l[8];
    return glm::max(irradiance, vec3{0.f});
}

using CubemapMipFacesF32 = std::array<DecodedImageF32, 6>;

CubemapMipFacesF32 BuildDiffuseCubemapFromIrradiance(const std::array<vec3, 9> &coefficients, float intensity, uint32_t size = 32u) {
    CubemapMipFacesF32 mip{};
    for (uint32_t face = 0; face < 6u; ++face) {
        auto &image = mip[face];
        image.Width = size;
        image.Height = size;
        image.Pixels.resize(size_t(size) * size * 4u, 1.f);
        for (uint32_t y = 0; y < size; ++y) {
            for (uint32_t x = 0; x < size; ++x) {
                const float u = 2.f * (float(x) + 0.5f) / float(size) - 1.f;
                const float v = 2.f * (float(y) + 0.5f) / float(size) - 1.f;
                const vec3 rgb = intensity * EvaluateIrradianceSH(coefficients, CubemapFaceDirection(face, u, v));
                const size_t offset = (size_t(y) * size + x) * 4u;
                image.Pixels[offset + 0] = rgb.r;
                image.Pixels[offset + 1] = rgb.g;
                image.Pixels[offset + 2] = rgb.b;
                image.Pixels[offset + 3] = 1.f;
            }
        }
    }
    return mip;
}

std::expected<CubemapEntry, std::string> CreateCubemapEntryFromMipFacesF32(
    const SceneVulkanResources &vk,
    mvk::BufferContext &ctx,
    vk::CommandPool command_pool,
    vk::Fence one_shot_fence,
    DescriptorSlots &slots,
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

    std::vector<float> pixels;
    std::vector<vk::BufferImageCopy> copies;
    size_t total_floats = 0;
    for (const auto &mip : mip_faces) {
        for (const auto &face : mip) {
            total_floats += face.Pixels.size();
        }
    }
    pixels.reserve(total_floats);
    copies.reserve(mip_faces.size() * 6u);

    size_t offset_bytes = 0;
    for (uint32_t mip = 0; mip < mip_faces.size(); ++mip) {
        const uint32_t size = std::max(1u, base_size >> mip);
        for (uint32_t face = 0; face < 6u; ++face) {
            const auto &src = mip_faces[mip][face].Pixels;
            copies.emplace_back(
                vk::BufferImageCopy{
                    offset_bytes,
                    0,
                    0,
                    vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, mip, face, 1},
                    {0, 0, 0},
                    {size, size, 1},
                }
            );
            pixels.insert(pixels.end(), src.begin(), src.end());
            offset_bytes += src.size() * sizeof(float);
        }
    }

    constexpr auto format = vk::Format::eR32G32B32A32Sfloat;
    auto image = mvk::CreateImage(
        vk.Device,
        vk.PhysicalDevice,
        {
            vk::ImageCreateFlagBits::eCubeCompatible,
            vk::ImageType::e2D,
            format,
            vk::Extent3D{base_size, base_size, 1},
            uint32_t(mip_faces.size()),
            6,
            vk::SampleCountFlagBits::e1,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
            vk::SharingMode::eExclusive,
        },
        {{}, {}, vk::ImageViewType::eCube, format, {}, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, uint32_t(mip_faces.size()), 0, 6}}
    );

    mvk::Buffer staging{ctx, as_bytes(std::span<const float>{pixels}), mvk::MemoryUsage::CpuOnly, vk::BufferUsageFlagBits::eTransferSrc};
    auto cb = std::move(vk.Device.allocateCommandBuffersUnique({command_pool, vk::CommandBufferLevel::ePrimary, 1}).front());
    cb->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    const vk::ImageSubresourceRange full_range{vk::ImageAspectFlagBits::eColor, 0, uint32_t(mip_faces.size()), 0, 6};
    cb->pipelineBarrier(
        vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eTransfer,
        {}, {}, {},
        vk::ImageMemoryBarrier{{}, vk::AccessFlagBits::eTransferWrite, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, {}, {}, *image.Image, full_range}
    );

    cb->copyBufferToImage(*staging, *image.Image, vk::ImageLayout::eTransferDstOptimal, copies);

    cb->pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eFragmentShader,
        {}, {}, {},
        vk::ImageMemoryBarrier{vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, {}, {}, *image.Image, full_range}
    );

    cb->end();
    Submit(vk.Queue, *cb, one_shot_fence, vk.Device);

    auto sampler = vk.Device.createSamplerUnique(
        vk::SamplerCreateInfo{
            {},
            vk::Filter::eLinear,
            vk::Filter::eLinear,
            vk::SamplerMipmapMode::eLinear,
            vk::SamplerAddressMode::eClampToEdge,
            vk::SamplerAddressMode::eClampToEdge,
            vk::SamplerAddressMode::eClampToEdge,
            0.f,
            VK_FALSE,
            1.f,
            VK_FALSE,
            vk::CompareOp::eNever,
            0.f,
            float(mip_faces.size()),
            vk::BorderColor::eIntOpaqueBlack,
            VK_FALSE,
        }
    );

    const auto sampler_slot = RegisterCubeSamplerSlot(slots, vk.Device, *sampler, *image.View);
    return CubemapEntry{.Image = std::move(image), .Sampler = std::move(sampler), .SamplerSlot = sampler_slot, .Size = base_size, .MipLevels = uint32_t(mip_faces.size()), .Name = std::move(name)};
}
} // namespace

uint32_t ComputeMipLevelCount(uint32_t width, uint32_t height) {
    const auto max_dim = std::max(width, height);
    return max_dim > 0 ? std::bit_width(max_dim) : 1u;
}

uint32_t RegisterCubeSamplerSlot(DescriptorSlots &slots, vk::Device device, vk::Sampler sampler, vk::ImageView image_view) {
    const auto slot = slots.Allocate(SlotType::CubeSampler);
    device.updateDescriptorSets({slots.MakeCubeSamplerWrite(slot, {sampler, image_view, vk::ImageLayout::eShaderReadOnlyOptimal})}, {});
    return slot;
}

std::vector<uint32_t> CollectSamplerSlots(std::span<const TextureEntry> textures) {
    std::vector<uint32_t> sampler_slots;
    sampler_slots.reserve(textures.size());
    for (const auto &texture : textures) {
        if (texture.SamplerSlot != InvalidSlot) sampler_slots.emplace_back(texture.SamplerSlot);
    }
    return sampler_slots;
}

void ReleaseSamplerSlots(DescriptorSlots &slots, std::span<const uint32_t> sampler_slots) {
    for (const auto sampler_slot : sampler_slots) {
        slots.Release({SlotType::Sampler, sampler_slot});
    }
}

void ReleaseCubeSamplerSlot(DescriptorSlots &slots, uint32_t sampler_slot) {
    if (sampler_slot == InvalidSlot) return;
    slots.Release({SlotType::CubeSampler, sampler_slot});
}

void ReleaseEnvironmentSamplerSlots(DescriptorSlots &slots, const EnvironmentStore &environments) {
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
    for (const auto *tex : {&environments.BrdfLut, &environments.SheenELut, &environments.CharlieLut}) {
        if (tex->SamplerSlot != InvalidSlot) slots.Release({SlotType::Sampler, tex->SamplerSlot});
    }
}

vk::SamplerAddressMode ToSamplerAddressMode(gltf::Wrap wrap) {
    switch (wrap) {
        case gltf::Wrap::ClampToEdge: return vk::SamplerAddressMode::eClampToEdge;
        case gltf::Wrap::MirroredRepeat: return vk::SamplerAddressMode::eMirroredRepeat;
        case gltf::Wrap::Repeat: return vk::SamplerAddressMode::eRepeat;
    }
    return vk::SamplerAddressMode::eRepeat;
}

SamplerConfig ToSamplerConfig(const gltf::Sampler *sampler) {
    SamplerConfig config{};
    if (sampler) {
        if (sampler->MagFilter && *sampler->MagFilter == gltf::Filter::Nearest) config.MagFilter = vk::Filter::eNearest;
        switch (sampler->MinFilter.value_or(gltf::Filter::LinearMipMapLinear)) {
            case gltf::Filter::Nearest:
                config.MinFilter = vk::Filter::eNearest;
                config.MipmapMode = vk::SamplerMipmapMode::eNearest;
                config.UsesMipmaps = false;
                break;
            case gltf::Filter::Linear:
                config.MinFilter = vk::Filter::eLinear;
                config.MipmapMode = vk::SamplerMipmapMode::eNearest;
                config.UsesMipmaps = false;
                break;
            case gltf::Filter::NearestMipMapNearest:
                config.MinFilter = vk::Filter::eNearest;
                config.MipmapMode = vk::SamplerMipmapMode::eNearest;
                config.UsesMipmaps = true;
                break;
            case gltf::Filter::LinearMipMapNearest:
                config.MinFilter = vk::Filter::eLinear;
                config.MipmapMode = vk::SamplerMipmapMode::eNearest;
                config.UsesMipmaps = true;
                break;
            case gltf::Filter::NearestMipMapLinear:
                config.MinFilter = vk::Filter::eNearest;
                config.MipmapMode = vk::SamplerMipmapMode::eLinear;
                config.UsesMipmaps = true;
                break;
            case gltf::Filter::LinearMipMapLinear:
                config.MinFilter = vk::Filter::eLinear;
                config.MipmapMode = vk::SamplerMipmapMode::eLinear;
                config.UsesMipmaps = true;
                break;
        }
    }
    return config;
}

TextureEntry CreateTextureEntry(
    const SceneVulkanResources &vk,
    mvk::BufferContext &ctx,
    vk::CommandPool command_pool,
    vk::Fence one_shot_fence,
    DescriptorSlots &slots,
    std::span<const std::byte> pixels_rgba8,
    uint32_t width,
    uint32_t height,
    std::string name,
    TextureColorSpace color_space,
    vk::SamplerAddressMode wrap_s,
    vk::SamplerAddressMode wrap_t,
    const SamplerConfig &sampler_cfg
) {
    const vk::Format texture_format = ToTextureFormat(color_space);
    const auto format_features = vk.PhysicalDevice.getFormatProperties(texture_format).optimalTilingFeatures;
    const bool supports_linear_blit = bool(format_features & vk::FormatFeatureFlagBits::eSampledImageFilterLinear);
    uint32_t mip_levels = sampler_cfg.UsesMipmaps && supports_linear_blit ? ComputeMipLevelCount(width, height) : 1u;
    if (mip_levels == 0) mip_levels = 1u;

    auto image = mvk::CreateImage(
        vk.Device,
        vk.PhysicalDevice,
        {
            {},
            vk::ImageType::e2D,
            texture_format,
            vk::Extent3D{width, height, 1},
            mip_levels,
            1,
            vk::SampleCountFlagBits::e1,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc,
            vk::SharingMode::eExclusive,
        },
        {{}, {}, vk::ImageViewType::e2D, texture_format, {}, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, mip_levels, 0, 1}}
    );

    mvk::Buffer staging{ctx, pixels_rgba8, mvk::MemoryUsage::CpuOnly, vk::BufferUsageFlagBits::eTransferSrc};
    auto cb = std::move(vk.Device.allocateCommandBuffersUnique({command_pool, vk::CommandBufferLevel::ePrimary, 1}).front());
    cb->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    const vk::ImageSubresourceRange full_range{vk::ImageAspectFlagBits::eColor, 0, mip_levels, 0, 1};
    cb->pipelineBarrier(
        vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eTransfer,
        {},
        {},
        {},
        vk::ImageMemoryBarrier{
            {},
            vk::AccessFlagBits::eTransferWrite,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eTransferDstOptimal,
            {},
            {},
            *image.Image,
            full_range,
        }
    );

    cb->copyBufferToImage(
        *staging,
        *image.Image,
        vk::ImageLayout::eTransferDstOptimal,
        vk::BufferImageCopy{
            0,
            0,
            0,
            vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 0, 0, 1},
            {0, 0, 0},
            {width, height, 1},
        }
    );

    int32_t mip_width = int32_t(width), mip_height = int32_t(height);
    for (uint32_t mip = 1; mip < mip_levels; ++mip) {
        cb->pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eTransfer,
            {}, {}, {},
            vk::ImageMemoryBarrier{
                vk::AccessFlagBits::eTransferWrite,
                vk::AccessFlagBits::eTransferRead,
                vk::ImageLayout::eTransferDstOptimal,
                vk::ImageLayout::eTransferSrcOptimal,
                {},
                {},
                *image.Image,
                vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, mip - 1, 1, 0, 1},
            }
        );

        cb->blitImage(
            *image.Image,
            vk::ImageLayout::eTransferSrcOptimal,
            *image.Image,
            vk::ImageLayout::eTransferDstOptimal,
            vk::ImageBlit{
                vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, mip - 1, 0, 1},
                {vk::Offset3D{0, 0, 0}, vk::Offset3D{mip_width, mip_height, 1}},
                vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, mip, 0, 1},
                {vk::Offset3D{0, 0, 0}, vk::Offset3D{std::max(1, mip_width / 2), std::max(1, mip_height / 2), 1}},
            },
            vk::Filter::eLinear
        );

        cb->pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eFragmentShader,
            {}, {}, {},
            vk::ImageMemoryBarrier{
                vk::AccessFlagBits::eTransferRead,
                vk::AccessFlagBits::eShaderRead,
                vk::ImageLayout::eTransferSrcOptimal,
                vk::ImageLayout::eShaderReadOnlyOptimal,
                {},
                {},
                *image.Image,
                vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, mip - 1, 1, 0, 1},
            }
        );

        mip_width = std::max(1, mip_width / 2);
        mip_height = std::max(1, mip_height / 2);
    }

    cb->pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eFragmentShader,
        {}, {}, {},
        vk::ImageMemoryBarrier{
            vk::AccessFlagBits::eTransferWrite,
            vk::AccessFlagBits::eShaderRead,
            vk::ImageLayout::eTransferDstOptimal,
            vk::ImageLayout::eShaderReadOnlyOptimal,
            {},
            {},
            *image.Image,
            vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, mip_levels - 1, 1, 0, 1},
        }
    );

    cb->end();
    Submit(vk.Queue, *cb, one_shot_fence, vk.Device);

    auto sampler = vk.Device.createSamplerUnique(
        vk::SamplerCreateInfo{
            {},
            sampler_cfg.MagFilter,
            sampler_cfg.MinFilter,
            sampler_cfg.MipmapMode,
            wrap_s,
            wrap_t,
            vk::SamplerAddressMode::eRepeat,
            0.f,
            VK_FALSE,
            1.f,
            VK_FALSE,
            vk::CompareOp::eNever,
            0.f,
            sampler_cfg.UsesMipmaps ? float(mip_levels) : 0.f,
            vk::BorderColor::eIntOpaqueBlack,
            VK_FALSE,
        }
    );

    const auto sampler_slot = slots.Allocate(SlotType::Sampler);
    const vk::DescriptorImageInfo sampler_info{*sampler, *image.View, vk::ImageLayout::eShaderReadOnlyOptimal};
    vk.Device.updateDescriptorSets({slots.MakeSamplerWrite(sampler_slot, sampler_info)}, {});

    return {.Image = std::move(image), .Sampler = std::move(sampler), .SamplerSlot = sampler_slot, .Width = width, .Height = height, .MipLevels = mip_levels, .Name = std::move(name)};
}

std::expected<TextureEntry, std::string> CreateTextureEntryFromEncoded(
    const SceneVulkanResources &vk,
    mvk::BufferContext &ctx,
    vk::CommandPool command_pool,
    vk::Fence one_shot_fence,
    DescriptorSlots &slots,
    std::span<const std::byte> encoded_bytes,
    std::string_view encoded_name,
    std::string texture_name,
    TextureColorSpace color_space,
    vk::SamplerAddressMode wrap_s,
    vk::SamplerAddressMode wrap_t,
    const SamplerConfig &sampler_cfg
) {
    auto decoded = DecodeImageRgba8(encoded_bytes, encoded_name);
    if (!decoded) return std::unexpected{std::move(decoded.error())};
    return CreateTextureEntry(vk, ctx, command_pool, one_shot_fence, slots, decoded->Pixels, decoded->Width, decoded->Height, std::move(texture_name), color_space, wrap_s, wrap_t, sampler_cfg);
}

std::expected<EnvironmentPrefiltered, std::string> CreateIblFromExtIbl(
    const SceneVulkanResources &vk,
    mvk::BufferContext &ctx,
    vk::CommandPool command_pool,
    vk::Fence one_shot_fence,
    DescriptorSlots &slots,
    const gltf::Scene &scene,
    const gltf::ImageBasedLight &ibl
) {
    std::vector<CubemapMipFacesF32> specular_mips;
    specular_mips.reserve(ibl.SpecularImageIndicesByMip.size());
    uint32_t specular_base_size = 0u;
    for (uint32_t mip = 0; mip < ibl.SpecularImageIndicesByMip.size(); ++mip) {
        CubemapMipFacesF32 faces{};
        for (uint32_t face = 0; face < 6u; ++face) {
            const auto image_index = ibl.SpecularImageIndicesByMip[mip][face];
            if (image_index >= scene.Images.size()) return std::unexpected{std::format("EXT_lights_image_based '{}' references image index {} (out of range).", ibl.Name, image_index)};

            const auto &src_image = scene.Images[image_index];
            auto decoded = DecodeImageRgba32f(
                src_image.Bytes,
                src_image.Name.empty() ? std::format("Image{}", image_index) : src_image.Name
            );
            if (!decoded) return std::unexpected{std::format("Failed to decode EXT_lights_image_based '{}' image {}: {}", ibl.Name, image_index, decoded.error())};
            if (decoded->Width != decoded->Height) return std::unexpected{std::format("EXT_lights_image_based '{}' image {} must be square (got {}x{}).", ibl.Name, image_index, decoded->Width, decoded->Height)};
            if (ibl.Intensity != 1.f) {
                for (auto &px : decoded->Pixels) px *= ibl.Intensity;
            }
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

        if (mip == 0u) specular_base_size = faces[0].Width;
        const uint32_t expected_size = std::max(1u, specular_base_size >> mip);
        if (faces[0].Width != expected_size) {
            return std::unexpected{std::format("EXT_lights_image_based '{}' mip {} has size {} but expected {}.", ibl.Name, mip, faces[0].Width, expected_size)};
        }
        specular_mips.emplace_back(std::move(faces));
    }

    auto specular_env = CreateCubemapEntryFromMipFacesF32(vk, ctx, command_pool, one_shot_fence, slots, specular_mips, ibl.Name + "_specular");
    if (!specular_env) return std::unexpected{std::move(specular_env.error())};

    std::vector<CubemapMipFacesF32> diffuse_mips;
    diffuse_mips.reserve(1);
    if (ibl.IrradianceCoefficients) diffuse_mips.emplace_back(BuildDiffuseCubemapFromIrradiance(*ibl.IrradianceCoefficients, ibl.Intensity));
    else diffuse_mips.emplace_back(specular_mips.back());

    auto diffuse_env = CreateCubemapEntryFromMipFacesF32(vk, ctx, command_pool, one_shot_fence, slots, diffuse_mips, ibl.Name + "_diffuse");
    if (!diffuse_env) {
        ReleaseCubeSamplerSlot(slots, specular_env->SamplerSlot);
        return std::unexpected{std::move(diffuse_env.error())};
    }

    return EnvironmentPrefiltered{
        .DiffuseEnv = std::move(*diffuse_env),
        .SpecularEnv = std::move(*specular_env),
        .Name = ibl.Name,
    };
}

// GPU-prefilters a Radiance HDR equirectangular image into a diffuse irradiance cubemap and a
// GGX specular prefiltered cubemap using dedicated compute pipelines. Returns the two bindless
// CubemapEntries. All temporary GPU resources (equirect image, raw cubemap, descriptor sets)
// are destroyed before returning.
EnvironmentPrefiltered CreateIblFromHdri(
    const SceneVulkanResources &vk,
    mvk::BufferContext &ctx,
    vk::CommandPool command_pool,
    vk::Fence fence,
    DescriptorSlots &slots,
    const IblPrefilterPipelines &prefilter,
    const std::filesystem::path &path,
    const std::string &name
) {
    // 1. Load HDR equirectangular image into a CPU staging buffer.
    const auto path_str = path.string();
    auto decoded = DecodeImageFileRgba32f(path, path_str);
    if (!decoded) throw std::runtime_error(std::format("Failed to load HDR '{}': {}", path_str, decoded.error()));
    const uint32_t eq_w = decoded->Width, eq_h = decoded->Height;

    const size_t eq_bytes = decoded->Pixels.size() * sizeof(float);
    mvk::Buffer eq_staging{
        ctx,
        std::span<const std::byte>{reinterpret_cast<const std::byte *>(decoded->Pixels.data()), eq_bytes},
        mvk::MemoryUsage::CpuOnly,
        vk::BufferUsageFlagBits::eTransferSrc
    };

    // 2. Upload equirect pixels to a temporary GPU image.
    constexpr auto rgba32f = vk::Format::eR32G32B32A32Sfloat;
    const vk::ImageSubresourceRange one_2d{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};

    auto equirect = mvk::CreateImage(
        vk.Device, vk.PhysicalDevice,
        {{}, vk::ImageType::e2D, rgba32f, vk::Extent3D{eq_w, eq_h, 1}, 1, 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst, vk::SharingMode::eExclusive},
        {{}, {}, vk::ImageViewType::e2D, rgba32f, {}, one_2d}
    );
    {
        auto cb = std::move(vk.Device.allocateCommandBuffersUnique({command_pool, vk::CommandBufferLevel::ePrimary, 1}).front());
        cb->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        cb->pipelineBarrier(
            vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer, {}, {}, {},
            vk::ImageMemoryBarrier{{}, vk::AccessFlagBits::eTransferWrite, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, {}, {}, *equirect.Image, one_2d}
        );
        cb->copyBufferToImage(*eq_staging, *equirect.Image, vk::ImageLayout::eTransferDstOptimal, vk::BufferImageCopy{0, 0, 0, vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 0, 0, 1}, {0, 0, 0}, {eq_w, eq_h, 1}});
        cb->pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, {}, {}, {},
            vk::ImageMemoryBarrier{vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, {}, {}, *equirect.Image, one_2d}
        );
        cb->end();
        Submit(vk.Queue, *cb, fence, vk.Device);
    }

    // 3. Create raw cubemap (512×512, full mip chain, storage+sampled+transfer).
    const uint32_t raw_size = 512;
    const uint32_t raw_mips = ComputeMipLevelCount(raw_size, raw_size);
    constexpr auto cube_flags = vk::ImageCreateFlagBits::eCubeCompatible;
    const vk::ImageSubresourceRange raw_full{vk::ImageAspectFlagBits::eColor, 0, raw_mips, 0, 6};

    auto raw_cube = mvk::CreateImage(
        vk.Device, vk.PhysicalDevice,
        {cube_flags, vk::ImageType::e2D, rgba32f, vk::Extent3D{raw_size, raw_size, 1}, raw_mips, 6,
         vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal,
         vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage |
             vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc,
         vk::SharingMode::eExclusive},
        {{}, {}, vk::ImageViewType::eCube, rgba32f, {}, raw_full}
    );
    // e2DArray view covering mip 0 of the raw cube — used as storage image write target.
    auto raw_cube_storage_view = vk.Device.createImageViewUnique(
        {{}, *raw_cube.Image, vk::ImageViewType::e2DArray, rgba32f, {}, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 6}}
    );

    // 4. Create diffuse irradiance cubemap (32×32, 1 mip).
    const uint32_t diff_size = 32;
    const vk::ImageSubresourceRange diff_range{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 6};
    auto diff_cube = mvk::CreateImage(
        vk.Device, vk.PhysicalDevice,
        {cube_flags, vk::ImageType::e2D, rgba32f, vk::Extent3D{diff_size, diff_size, 1}, 1, 6,
         vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal,
         vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage |
             vk::ImageUsageFlagBits::eTransferDst,
         vk::SharingMode::eExclusive},
        {{}, {}, vk::ImageViewType::eCube, rgba32f, {}, diff_range}
    );
    auto diff_storage_view = vk.Device.createImageViewUnique(
        {{}, *diff_cube.Image, vk::ImageViewType::e2DArray, rgba32f, {}, diff_range}
    );

    // 5. Create specular prefiltered cubemap (256×256, full mip chain).
    const uint32_t spec_size = 256;
    const uint32_t spec_mips = ComputeMipLevelCount(spec_size, spec_size);
    const vk::ImageSubresourceRange spec_full{vk::ImageAspectFlagBits::eColor, 0, spec_mips, 0, 6};
    auto spec_cube = mvk::CreateImage(
        vk.Device, vk.PhysicalDevice,
        {cube_flags, vk::ImageType::e2D, rgba32f, vk::Extent3D{spec_size, spec_size, 1}, spec_mips, 6,
         vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal,
         vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage |
             vk::ImageUsageFlagBits::eTransferDst,
         vk::SharingMode::eExclusive},
        {{}, {}, vk::ImageViewType::eCube, rgba32f, {}, spec_full}
    );
    // One e2DArray storage view per specular mip level.
    std::vector<vk::UniqueImageView> spec_storage_views;
    spec_storage_views.reserve(spec_mips);
    for (uint32_t mip = 0; mip < spec_mips; ++mip) {
        spec_storage_views.push_back(vk.Device.createImageViewUnique(
            {{}, *spec_cube.Image, vk::ImageViewType::e2DArray, rgba32f, {}, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, mip, 1, 0, 6}}
        ));
    }

    // 6. Allocate local descriptor sets (one per dispatch variant).
    //    Layout is owned by IblPrefilterPipelines; we only create the pool and sets.
    const uint32_t num_sets = 2u + spec_mips; // equirect→cube, diffuse, specular×mips
    const std::array pool_sizes{
        vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler, num_sets},
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, num_sets},
    };
    auto desc_pool = vk.Device.createDescriptorPoolUnique({{}, num_sets, pool_sizes});
    const std::vector<vk::DescriptorSetLayout> layouts(num_sets, *prefilter.DescriptorSetLayout);
    auto desc_sets = vk.Device.allocateDescriptorSets({*desc_pool, layouts});
    // desc_sets[0]       = EquirectToCubemap (equirect in, raw cube mip0 out)
    // desc_sets[1]       = DiffuseIrradiance (raw cube in, diffuse out)
    // desc_sets[2+mip]   = SpecularPrefilter (raw cube in, specular mip N out)

    // 7. Update all descriptor sets before recording.
    const vk::SamplerCreateInfo linear_clamp_ci{
        {},
        vk::Filter::eLinear,
        vk::Filter::eLinear,
        vk::SamplerMipmapMode::eLinear,
        vk::SamplerAddressMode::eClampToEdge,
        vk::SamplerAddressMode::eClampToEdge,
        vk::SamplerAddressMode::eClampToEdge,
        0.f,
        VK_FALSE,
        1.f,
        VK_FALSE,
        vk::CompareOp::eNever,
        0.f,
        1000.f,
        vk::BorderColor::eIntOpaqueBlack,
        VK_FALSE,
    };
    const vk::SamplerCreateInfo linear_repeat_ci{
        {},
        vk::Filter::eLinear,
        vk::Filter::eLinear,
        vk::SamplerMipmapMode::eLinear,
        vk::SamplerAddressMode::eRepeat,
        vk::SamplerAddressMode::eRepeat,
        vk::SamplerAddressMode::eRepeat,
        0.f,
        VK_FALSE,
        1.f,
        VK_FALSE,
        vk::CompareOp::eNever,
        0.f,
        1000.f,
        vk::BorderColor::eIntOpaqueBlack,
        VK_FALSE,
    };
    auto equirect_sampler = vk.Device.createSamplerUnique(linear_repeat_ci);
    auto raw_cube_sampler = vk.Device.createSamplerUnique(linear_clamp_ci);

    const vk::DescriptorImageInfo eq_info{*equirect_sampler, *equirect.View, vk::ImageLayout::eShaderReadOnlyOptimal};
    const vk::DescriptorImageInfo raw_mip0_info{{}, *raw_cube_storage_view, vk::ImageLayout::eGeneral};
    vk.Device.updateDescriptorSets({
                                       vk::WriteDescriptorSet{desc_sets[0], 0, 0, 1, vk::DescriptorType::eCombinedImageSampler, &eq_info},
                                       vk::WriteDescriptorSet{desc_sets[0], 1, 0, 1, vk::DescriptorType::eStorageImage, &raw_mip0_info},
                                   },
                                   {});

    const vk::DescriptorImageInfo raw_cube_info{*raw_cube_sampler, *raw_cube.View, vk::ImageLayout::eShaderReadOnlyOptimal};
    const vk::DescriptorImageInfo diff_info{{}, *diff_storage_view, vk::ImageLayout::eGeneral};
    vk.Device.updateDescriptorSets({
                                       vk::WriteDescriptorSet{desc_sets[1], 0, 0, 1, vk::DescriptorType::eCombinedImageSampler, &raw_cube_info},
                                       vk::WriteDescriptorSet{desc_sets[1], 1, 0, 1, vk::DescriptorType::eStorageImage, &diff_info},
                                   },
                                   {});

    for (uint32_t mip = 0; mip < spec_mips; ++mip) {
        const vk::DescriptorImageInfo spec_mip_info{{}, *spec_storage_views[mip], vk::ImageLayout::eGeneral};
        vk.Device.updateDescriptorSets({
                                           vk::WriteDescriptorSet{desc_sets[2 + mip], 0, 0, 1, vk::DescriptorType::eCombinedImageSampler, &raw_cube_info},
                                           vk::WriteDescriptorSet{desc_sets[2 + mip], 1, 0, 1, vk::DescriptorType::eStorageImage, &spec_mip_info},
                                       },
                                       {});
    }

    // 8. Record and submit the full prefiltering command buffer.
    auto cb = std::move(vk.Device.allocateCommandBuffersUnique({command_pool, vk::CommandBufferLevel::ePrimary, 1}).front());
    cb->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    // --- Initial layout transitions ---
    // raw cube mip 0: Undefined → General (storage write by EquirectToCubemap)
    cb->pipelineBarrier(
        vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eComputeShader, {}, {}, {},
        vk::ImageMemoryBarrier{{}, vk::AccessFlagBits::eShaderWrite, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral, {}, {}, *raw_cube.Image, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 6}}
    );
    // raw cube mips 1..N: Undefined → TransferDstOptimal (blit targets for mipmap generation)
    if (raw_mips > 1) {
        cb->pipelineBarrier(
            vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer, {}, {}, {},
            vk::ImageMemoryBarrier{{}, vk::AccessFlagBits::eTransferWrite, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, {}, {}, *raw_cube.Image, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 1, raw_mips - 1, 0, 6}}
        );
    }
    // diffuse: Undefined → General
    cb->pipelineBarrier(
        vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eComputeShader, {}, {}, {},
        vk::ImageMemoryBarrier{{}, vk::AccessFlagBits::eShaderWrite, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral, {}, {}, *diff_cube.Image, diff_range}
    );
    // specular all mips: Undefined → General
    cb->pipelineBarrier(
        vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eComputeShader, {}, {}, {},
        vk::ImageMemoryBarrier{{}, vk::AccessFlagBits::eShaderWrite, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral, {}, {}, *spec_cube.Image, spec_full}
    );

    // --- EquirectToCubemap pass ---
    cb->bindPipeline(vk::PipelineBindPoint::eCompute, *prefilter.EquirectToCubemap);
    cb->bindDescriptorSets(vk::PipelineBindPoint::eCompute, *prefilter.PipelineLayout, 0, desc_sets[0], {});
    cb->pushConstants(*prefilter.PipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(uint32_t), &raw_size);
    cb->dispatch((raw_size + 7) / 8, (raw_size + 7) / 8, 6);

    // raw cube mip 0: General → TransferSrcOptimal (source for mipmap blit chain)
    cb->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, {}, {}, {},
        vk::ImageMemoryBarrier{vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eTransferRead, vk::ImageLayout::eGeneral, vk::ImageLayout::eTransferSrcOptimal, {}, {}, *raw_cube.Image, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 6}}
    );

    // --- Generate mip chain for raw cubemap ---
    int32_t mip_size = int32_t(raw_size);
    for (uint32_t mip = 1; mip < raw_mips; ++mip) {
        const int32_t next_size = std::max(1, mip_size / 2);
        // Blit all 6 faces: mip N-1 (TransferSrcOptimal) → mip N (TransferDstOptimal)
        cb->blitImage(
            *raw_cube.Image, vk::ImageLayout::eTransferSrcOptimal,
            *raw_cube.Image, vk::ImageLayout::eTransferDstOptimal,
            vk::ImageBlit{
                vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, mip - 1, 0, 6},
                {vk::Offset3D{0, 0, 0}, vk::Offset3D{mip_size, mip_size, 1}},
                vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, mip, 0, 6},
                {vk::Offset3D{0, 0, 0}, vk::Offset3D{next_size, next_size, 1}},
            },
            vk::Filter::eLinear
        );
        // mip N-1: TransferSrcOptimal → ShaderReadOnlyOptimal (done as blit source)
        cb->pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, {}, {}, {},
            vk::ImageMemoryBarrier{vk::AccessFlagBits::eTransferRead, vk::AccessFlagBits::eShaderRead, vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, {}, {}, *raw_cube.Image, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, mip - 1, 1, 0, 6}}
        );
        if (mip < raw_mips - 1) {
            // mip N: TransferDstOptimal → TransferSrcOptimal (source for next blit)
            cb->pipelineBarrier(
                vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer, {}, {}, {},
                vk::ImageMemoryBarrier{vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eTransferRead, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eTransferSrcOptimal, {}, {}, *raw_cube.Image, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, mip, 1, 0, 6}}
            );
        } else {
            // Last mip: TransferDstOptimal → ShaderReadOnlyOptimal
            cb->pipelineBarrier(
                vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, {}, {}, {},
                vk::ImageMemoryBarrier{vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, {}, {}, *raw_cube.Image, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, mip, 1, 0, 6}}
            );
        }
        mip_size = next_size;
    }

    // --- DiffuseIrradiance pass ---
    cb->bindPipeline(vk::PipelineBindPoint::eCompute, *prefilter.DiffuseIrradiance);
    cb->bindDescriptorSets(vk::PipelineBindPoint::eCompute, *prefilter.PipelineLayout, 0, desc_sets[1], {});
    cb->pushConstants(*prefilter.PipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(uint32_t), &diff_size);
    cb->dispatch((diff_size + 7) / 8, (diff_size + 7) / 8, 6);

    // diffuse: General → ShaderReadOnlyOptimal
    cb->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {},
        vk::ImageMemoryBarrier{vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead, vk::ImageLayout::eGeneral, vk::ImageLayout::eShaderReadOnlyOptimal, {}, {}, *diff_cube.Image, diff_range}
    );

    // --- SpecularPrefilter passes (one per roughness mip) ---
    cb->bindPipeline(vk::PipelineBindPoint::eCompute, *prefilter.SpecularPrefilter);
    for (uint32_t mip = 0; mip < spec_mips; ++mip) {
        const uint32_t mip_face_size = std::max(1u, spec_size >> mip);
        struct SpecPC {
            uint32_t FaceSize;
            uint32_t SourceSize;
            float Roughness;
        };
        const SpecPC pc{
            .FaceSize = mip_face_size,
            .SourceSize = raw_size,
            .Roughness = float(mip) / float(spec_mips - 1),
        };
        cb->bindDescriptorSets(vk::PipelineBindPoint::eCompute, *prefilter.PipelineLayout, 0, desc_sets[2 + mip], {});
        cb->pushConstants(*prefilter.PipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(SpecPC), &pc);
        cb->dispatch((mip_face_size + 7) / 8, (mip_face_size + 7) / 8, 6);
    }

    // specular: General → ShaderReadOnlyOptimal
    cb->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {},
        vk::ImageMemoryBarrier{vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead, vk::ImageLayout::eGeneral, vk::ImageLayout::eShaderReadOnlyOptimal, {}, {}, *spec_cube.Image, spec_full}
    );

    cb->end();
    Submit(vk.Queue, *cb, fence, vk.Device);
    // desc_pool destroyed here, freeing desc_sets. equirect, raw_cube, storage views, local samplers also destroyed.

    // 9. Register diffuse and specular cubemaps in the global bindless CubeSamplers array.
    auto diff_sampler = vk.Device.createSamplerUnique(linear_clamp_ci);
    const uint32_t diff_slot = RegisterCubeSamplerSlot(slots, vk.Device, *diff_sampler, *diff_cube.View);

    auto spec_sampler = vk.Device.createSamplerUnique(vk::SamplerCreateInfo{
        {},
        vk::Filter::eLinear,
        vk::Filter::eLinear,
        vk::SamplerMipmapMode::eLinear,
        vk::SamplerAddressMode::eClampToEdge,
        vk::SamplerAddressMode::eClampToEdge,
        vk::SamplerAddressMode::eClampToEdge,
        0.f,
        VK_FALSE,
        1.f,
        VK_FALSE,
        vk::CompareOp::eNever,
        0.f,
        float(spec_mips),
        vk::BorderColor::eIntOpaqueBlack,
        VK_FALSE,
    });
    const uint32_t spec_slot = RegisterCubeSamplerSlot(slots, vk.Device, *spec_sampler, *spec_cube.View);

    return {
        .DiffuseEnv = {.Image = std::move(diff_cube), .Sampler = std::move(diff_sampler), .SamplerSlot = diff_slot, .Size = diff_size, .MipLevels = 1, .Name = name + "_diffuse"},
        .SpecularEnv = {.Image = std::move(spec_cube), .Sampler = std::move(spec_sampler), .SamplerSlot = spec_slot, .Size = spec_size, .MipLevels = spec_mips, .Name = name + "_specular"},
        .Name = name,
    };
}

IblSamplers MakeIblSamplers(const EnvironmentPrefiltered &pre, const EnvironmentStore &environments) {
    return {
        .DiffuseEnvSamplerSlot = pre.DiffuseEnv.SamplerSlot,
        .SpecularEnvSamplerSlot = pre.SpecularEnv.SamplerSlot,
        .BrdfLutSamplerSlot = environments.BrdfLut.SamplerSlot,
        .SpecularEnvMipCount = pre.SpecularEnv.MipLevels,
        .SheenEnvSamplerSlot = pre.SpecularEnv.SamplerSlot,
        .SheenEnvMipCount = pre.SpecularEnv.MipLevels,
        .SheenELutSamplerSlot = environments.SheenELut.SamplerSlot,
        .CharlieLutSamplerSlot = environments.CharlieLut.SamplerSlot,
    };
}

TextureEntry CreateDefaultLutTexture(const SceneVulkanResources &vk, mvk::BufferContext &ctx, vk::CommandPool command_pool, vk::Fence one_shot_fence, DescriptorSlots &slots, std::string_view lut_path, std::string_view name) {
    const auto encoded = File::Read(std::filesystem::path{lut_path});
    auto texture = CreateTextureEntryFromEncoded(vk, ctx, command_pool, one_shot_fence, slots, std::as_bytes(std::span{encoded}), lut_path, std::string{name}, TextureColorSpace::Linear, vk::SamplerAddressMode::eClampToEdge, vk::SamplerAddressMode::eClampToEdge, SamplerConfig{.UsesMipmaps = false});
    if (!texture) throw std::runtime_error(std::format("Failed to initialize default LUT texture '{}': {}", lut_path, texture.error()));
    return std::move(*texture);
}
