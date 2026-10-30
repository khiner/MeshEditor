#include "metal/Image.h"
#include "metal/AutoreleaseScope.h"
#include "metal/PassChain.h"

#include "metal/MetalCpp.h"

namespace mtl {
namespace {
MTL::StorageMode StorageFor(MTL::TextureUsage usage) {
    return (usage & MTL::TextureUsageRenderTarget) != 0 ? MTL::StorageModePrivate : MTL::StorageModeShared;
}

Texture Create(
    const Context &ctx, MTL::TextureType type, MTL::PixelFormat format, Extent2D extent,
    MTL::TextureUsage usage, uint32_t mip_levels, MTL::StorageMode storage, uint32_t layers = 1
) {
    const AutoreleaseScope pool;
    const auto descriptor = NS::TransferPtr(MTL::TextureDescriptor::alloc()->init());
    descriptor->setTextureType(type);
    descriptor->setPixelFormat(format);
    descriptor->setWidth(extent.Width);
    descriptor->setHeight(extent.Height);
    if (layers > 1) descriptor->setArrayLength(layers);
    descriptor->setMipmapLevelCount(mip_levels);
    descriptor->setUsage(usage);
    descriptor->setStorageMode(storage);
    auto handle = NS::TransferPtr(ctx.Device->newTexture(descriptor.get()));
    if (!handle) throw std::runtime_error("Failed to allocate a Metal texture.");
    return {ctx, std::move(handle), extent, mip_levels};
}

// Runs the copy after every earlier command and returns once it completes.
bool Copy(const Context &ctx, auto &&encode) {
    const AutoreleaseScope pool;
    auto command = ctx.Begin();
    {
        PassChain chain{command};
        encode(chain.BeginBlit());
    }
    return ctx.Run(std::move(command)).Error.empty();
}
} // namespace

Texture CreateTexture2D(const Context &ctx, MTL::PixelFormat format, Extent2D extent, MTL::TextureUsage usage, uint32_t mip_levels, std::optional<MTL::StorageMode> storage) {
    return Create(ctx, MTL::TextureType2D, format, extent, usage, mip_levels, storage.value_or(StorageFor(usage)));
}

Texture CreateTextureCube(const Context &ctx, MTL::PixelFormat format, uint32_t size, MTL::TextureUsage usage, uint32_t mip_levels) {
    return Create(ctx, MTL::TextureTypeCube, format, {size, size}, usage, mip_levels, StorageFor(usage));
}

Texture MipView(const Texture &texture, uint32_t mip, MTL::TextureType type, uint32_t layers) {
    const AutoreleaseScope pool;
    auto handle = NS::TransferPtr(texture.Handle->newTextureView(texture.Handle->pixelFormat(), type, NS::Range::Make(mip, 1), NS::Range::Make(0, layers)));
    return {*texture.Handle.Ctx, std::move(handle), {std::max(texture.Extent.Width >> mip, 1u), std::max(texture.Extent.Height >> mip, 1u)}};
}

Texture CreateMipView(const Texture &texture, uint32_t mip) { return MipView(texture, mip, texture.Handle->textureType(), uint32_t(texture.Handle->arrayLength())); }
Texture CreateCubeMipView(const Texture &texture, uint32_t mip) { return MipView(texture, mip, MTL::TextureType2DArray, 6u); }

NS::SharedPtr<MTL::SamplerState> CreateSampler(const Context &ctx, const SamplerDesc &desc) {
    const AutoreleaseScope pool;
    const auto descriptor = NS::TransferPtr(MTL::SamplerDescriptor::alloc()->init());
    descriptor->setMinFilter(desc.MinFilter);
    descriptor->setMagFilter(desc.MagFilter);
    descriptor->setMipFilter(desc.MipFilter);
    descriptor->setSAddressMode(desc.AddressS);
    descriptor->setTAddressMode(desc.AddressT);
    descriptor->setRAddressMode(desc.AddressR);
    descriptor->setMaxAnisotropy(NS::UInteger(desc.MaxAnisotropy));
    // Argument tables and argument buffers bind samplers by resource id.
    descriptor->setSupportArgumentBuffers(true);
    auto sampler = NS::TransferPtr(ctx.Device->newSamplerState(descriptor.get()));
    if (!sampler) throw std::runtime_error("Failed to create a Metal sampler.");
    return sampler;
}

NS::SharedPtr<MTL::SamplerState> CreateSampler(
    const Context &ctx, MTL::SamplerMinMagFilter filter, MTL::SamplerMipFilter mip_filter,
    MTL::SamplerAddressMode address_mode, float max_anisotropy
) {
    return CreateSampler(ctx, {filter, filter, mip_filter, address_mode, address_mode, address_mode, max_anisotropy});
}

bool CopyTextureRegion(const Context &ctx, const Texture &source, uint32_t x, uint32_t y, Extent2D extent, const Texture &destination) {
    return Copy(ctx, [&](ComputeEncoder encoder) {
        encoder->copyFromTexture(
            *source, 0, 0, MTL::Origin(x, y, 0), MTL::Size(extent.Width, extent.Height, 1),
            *destination, 0, 0, MTL::Origin(0, 0, 0)
        );
    });
}

bool CopyTextureRegion(const Context &ctx, const Texture &source, uint32_t x, uint32_t y, Extent2D extent, MTL::Buffer *destination, uint32_t bytes_per_row) {
    return Copy(ctx, [&](ComputeEncoder encoder) {
        encoder->copyFromTexture(
            *source, 0, 0, MTL::Origin(x, y, 0), MTL::Size(extent.Width, extent.Height, 1),
            destination, 0, bytes_per_row, uint64_t(bytes_per_row) * extent.Height
        );
    });
}

void Upload(const Texture &texture, uint32_t mip, std::span<const std::byte> bytes, uint32_t bytes_per_row, uint32_t layer) {
    const AutoreleaseScope pool;
    const auto width = std::max(texture.Extent.Width >> mip, 1u);
    const auto height = std::max(texture.Extent.Height >> mip, 1u);
    texture.Handle->replaceRegion(MTL::Region::Make2D(0, 0, width, height), mip, layer, bytes.data(), bytes_per_row, 0);
}
} // namespace mtl
