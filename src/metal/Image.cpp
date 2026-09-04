#include "metal/Image.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace mtl {
namespace {
MTL::StorageMode StorageFor(MTL::TextureUsage usage) {
    return (usage & MTL::TextureUsageRenderTarget) != 0 ? MTL::StorageModePrivate : MTL::StorageModeShared;
}

Texture Create(
    const Context &ctx, MTL::TextureType type, MTL::PixelFormat format, Extent2D extent,
    MTL::TextureUsage usage, uint32_t mip_levels, MTL::StorageMode storage,
    uint32_t layers = 1, bool track_residency = true
) {
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
    return {ctx, std::move(handle), extent, mip_levels, track_residency};
}

bool Blit(const Context &ctx, auto &&encode) {
    auto *command_buffer = ctx.Queue->commandBuffer();
    auto *blit = command_buffer->blitCommandEncoder();
    encode(blit);
    blit->endEncoding();
    command_buffer->commit();
    command_buffer->waitUntilCompleted();
    return command_buffer->error() == nullptr;
}
} // namespace

Texture::Texture(const Context &ctx, NS::SharedPtr<MTL::Texture> handle, Extent2D extent, uint32_t mip_levels, bool track_residency)
    : Handle{std::move(handle)}, Extent{extent}, MipLevels{mip_levels}, ResidencyContext{track_residency ? &ctx : nullptr} {
    if (ResidencyContext) ResidencyContext->AddResident(Handle.get());
}

Texture::Texture(Texture &&other) noexcept
    : Handle{std::move(other.Handle)}, Extent{other.Extent}, MipLevels{other.MipLevels},
      ResidencyContext{std::exchange(other.ResidencyContext, nullptr)} {}

Texture &Texture::operator=(Texture &&other) noexcept {
    if (this == &other) return *this;
    if (ResidencyContext && Handle) ResidencyContext->RemoveResident(Handle.get());
    Handle = std::move(other.Handle);
    Extent = other.Extent;
    MipLevels = other.MipLevels;
    ResidencyContext = std::exchange(other.ResidencyContext, nullptr);
    return *this;
}

Texture::~Texture() {
    if (ResidencyContext && Handle) ResidencyContext->RemoveResident(Handle.get());
}

Texture CreateTexture2D(const Context &ctx, MTL::PixelFormat format, Extent2D extent, MTL::TextureUsage usage, uint32_t mip_levels, std::optional<MTL::StorageMode> storage) {
    return Create(ctx, MTL::TextureType2D, format, extent, usage, mip_levels, storage.value_or(StorageFor(usage)));
}

Texture CreateUntrackedTexture2D(const Context &ctx, MTL::PixelFormat format, Extent2D extent, MTL::TextureUsage usage, std::optional<MTL::StorageMode> storage) {
    return Create(ctx, MTL::TextureType2D, format, extent, usage, 1, storage.value_or(StorageFor(usage)), 1, false);
}

Texture CreateTextureCube(const Context &ctx, MTL::PixelFormat format, uint32_t size, MTL::TextureUsage usage, uint32_t mip_levels) {
    return Create(ctx, MTL::TextureTypeCube, format, {size, size}, usage, mip_levels, StorageFor(usage));
}

Texture CreateTexture2DArray(const Context &ctx, MTL::PixelFormat format, Extent2D extent, uint32_t layers, MTL::TextureUsage usage, uint32_t mip_levels) {
    return Create(ctx, MTL::TextureType2DArray, format, extent, usage, mip_levels, StorageFor(usage), layers);
}

Texture CreateMipView(const Texture &texture, uint32_t mip) {
    auto handle = NS::TransferPtr(texture.Handle->newTextureView(
        texture.Handle->pixelFormat(), texture.Handle->textureType(),
        NS::Range::Make(mip, 1), NS::Range::Make(0, texture.Handle->arrayLength())
    ));
    return {*texture.ResidencyContext, std::move(handle), {std::max(texture.Extent.Width >> mip, 1u), std::max(texture.Extent.Height >> mip, 1u)}};
}

Texture CreateCubeMipView(const Texture &texture, uint32_t mip) {
    auto handle = NS::TransferPtr(texture.Handle->newTextureView(
        texture.Handle->pixelFormat(), MTL::TextureType2DArray, NS::Range::Make(mip, 1), NS::Range::Make(0, 6)
    ));
    return {*texture.ResidencyContext, std::move(handle), {std::max(texture.Extent.Width >> mip, 1u), std::max(texture.Extent.Height >> mip, 1u)}};
}

NS::SharedPtr<MTL::SamplerState> CreateSampler(const Context &ctx, const SamplerDesc &desc) {
    const auto descriptor = NS::TransferPtr(MTL::SamplerDescriptor::alloc()->init());
    descriptor->setMinFilter(desc.MinFilter);
    descriptor->setMagFilter(desc.MagFilter);
    descriptor->setMipFilter(desc.MipFilter);
    descriptor->setSAddressMode(desc.AddressS);
    descriptor->setTAddressMode(desc.AddressT);
    descriptor->setRAddressMode(desc.AddressR);
    descriptor->setMaxAnisotropy(NS::UInteger(desc.MaxAnisotropy));
    // Argument-buffer samplers require resource IDs.
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
    return Blit(ctx, [&](MTL::BlitCommandEncoder *blit) {
        blit->copyFromTexture(
            *source, 0, 0, MTL::Origin(x, y, 0), MTL::Size(extent.Width, extent.Height, 1),
            *destination, 0, 0, MTL::Origin(0, 0, 0)
        );
    });
}

bool CopyTextureRegion(const Context &ctx, const Texture &source, uint32_t x, uint32_t y, Extent2D extent, MTL::Buffer *destination, uint32_t bytes_per_row) {
    return Blit(ctx, [&](MTL::BlitCommandEncoder *blit) {
        blit->copyFromTexture(
            *source, 0, 0, MTL::Origin(x, y, 0), MTL::Size(extent.Width, extent.Height, 1),
            destination, 0, bytes_per_row, uint64_t(bytes_per_row) * extent.Height
        );
    });
}

void Upload(const Texture &texture, uint32_t mip, std::span<const std::byte> bytes, uint32_t bytes_per_row, uint32_t layer) {
    const auto width = std::max(texture.Extent.Width >> mip, 1u);
    const auto height = std::max(texture.Extent.Height >> mip, 1u);
    texture.Handle->replaceRegion(MTL::Region::Make2D(0, 0, width, height), mip, layer, bytes.data(), bytes_per_row, 0);
}
} // namespace mtl
