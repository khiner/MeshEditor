#pragma once

#include "metal/MetalContext.h"

#include <bit>
#include <cstdint>
#include <optional>
#include <span>

namespace mtl {
namespace Format {
inline constexpr auto Color = MTL::PixelFormatBGRA8Unorm;
inline constexpr auto HdrColor = MTL::PixelFormatRGBA16Float;
inline constexpr auto Depth = MTL::PixelFormatDepth32Float;
inline constexpr auto Float = MTL::PixelFormatR32Float;
inline constexpr auto Float2 = MTL::PixelFormatRG32Float;
inline constexpr auto LineData = MTL::PixelFormatRGBA8Unorm;
inline constexpr auto Velocity = MTL::PixelFormatRGBA16Float;
inline constexpr auto Uint = MTL::PixelFormatR32Uint;
}

struct Extent2D {
    uint32_t Width, Height;
    bool operator==(const Extent2D &) const = default;
};

struct Texture {
    NS::SharedPtr<MTL::Texture> Handle;
    Extent2D Extent{};
    uint32_t MipLevels{1};

    MTL::Texture *operator*() const { return Handle.get(); }
    explicit operator bool() const { return bool(Handle); }
};

// `storage` disambiguates CPU-visible textures from private attachments with the same usage.
Texture CreateTexture2D(const Context &, MTL::PixelFormat, Extent2D, MTL::TextureUsage, uint32_t mip_levels = 1, std::optional<MTL::StorageMode> storage = {});
Texture CreateTextureCube(const Context &, MTL::PixelFormat, uint32_t size, MTL::TextureUsage, uint32_t mip_levels = 1);
Texture CreateTexture2DArray(const Context &, MTL::PixelFormat, Extent2D, uint32_t layers, MTL::TextureUsage, uint32_t mip_levels = 1);
NS::SharedPtr<MTL::Texture> CreateMipView(const Texture &, uint32_t mip);
NS::SharedPtr<MTL::Texture> CreateCubeMipView(const Texture &, uint32_t mip);

constexpr uint32_t MipLevelCount(uint32_t width, uint32_t height) {
    const auto max_dim = std::max(width, height);
    return max_dim > 0 ? uint32_t(std::bit_width(max_dim)) : 1u;
}

struct SamplerDesc {
    MTL::SamplerMinMagFilter MinFilter{MTL::SamplerMinMagFilterNearest}, MagFilter{MTL::SamplerMinMagFilterNearest};
    MTL::SamplerMipFilter MipFilter{MTL::SamplerMipFilterNotMipmapped};
    MTL::SamplerAddressMode AddressS{MTL::SamplerAddressModeClampToEdge}, AddressT{MTL::SamplerAddressModeClampToEdge}, AddressR{MTL::SamplerAddressModeClampToEdge};
    float MaxAnisotropy{1.f};
};

NS::SharedPtr<MTL::SamplerState> CreateSampler(const Context &, const SamplerDesc &);
NS::SharedPtr<MTL::SamplerState> CreateSampler(const Context &, MTL::SamplerMinMagFilter, MTL::SamplerMipFilter, MTL::SamplerAddressMode, float max_anisotropy = 1.f);

// Synchronous blits from a private texture. Return false on copy failure.
bool CopyTextureRegion(const Context &, const Texture &source, uint32_t x, uint32_t y, Extent2D, const Texture &destination);
bool CopyTextureRegion(const Context &, const Texture &source, uint32_t x, uint32_t y, Extent2D, MTL::Buffer *destination, uint32_t bytes_per_row);

void Upload(const Texture &, uint32_t mip, std::span<const std::byte> bytes, uint32_t bytes_per_row, uint32_t layer = 0);
}
