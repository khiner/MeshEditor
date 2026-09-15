#include "render/RenderTargets.h"

#include "metal/Bindless.h"
#include "metal/Buffer.h"

#include <bit>

namespace Format = mtl::Format;

RenderTargets::ResourcesT::ResourcesT(const mtl::Context &ctx, mtl::Extent2D extent, mtl::BindlessSet &slots)
    // Visibility depth remains paired with its IDs throughout shading and selection.
    : VisibilityDepth{mtl::CreateTexture2D(ctx, Format::Depth, extent, MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead)},
      ScratchDepth{mtl::CreateTexture2D(ctx, Format::Depth, extent, MTL::TextureUsageRenderTarget)},
      VisibilityImage{mtl::CreateTexture2D(ctx, Format::Uint, extent, MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead)},
      SilhouetteImage{mtl::CreateTexture2D(ctx, Format::Float2, extent, MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead)},
      SceneColorImage{mtl::CreateTexture2D(ctx, Format::HdrColor, extent, MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead)},
      OverlayColorImage{mtl::CreateTexture2D(ctx, Format::Color, extent, MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead)},
      FinalColorImage{mtl::CreateTexture2D(ctx, Format::Color, extent, MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead)},
      DepthPyramidImage{[&] {
          const mtl::Extent2D padded{std::bit_ceil((extent.Width + 1) / 2), std::bit_ceil((extent.Height + 1) / 2)};
          return mtl::CreateTexture2D(ctx, Format::Float, padded, MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite, mtl::MipLevelCount(padded.Width, padded.Height));
      }()},
      DepthPyramidMips{[&] {
          std::vector<PyramidMip> mips;
          mips.reserve(DepthPyramidImage.MipLevels);
          for (uint32_t mip = 0; mip < DepthPyramidImage.MipLevels; ++mip) {
              const mtl::Extent2D data_extent{((extent.Width - 1) >> (mip + 1)) + 1, ((extent.Height - 1) >> (mip + 1)) + 1};
              mips.push_back({mtl::CreateMipView(DepthPyramidImage, mip), slots.Allocate(SlotType::Image), data_extent});
          }
          return mips;
      }()},
      NearestSampler{mtl::CreateSampler(ctx, MTL::SamplerMinMagFilterNearest, MTL::SamplerMipFilterNearest, MTL::SamplerAddressModeClampToEdge)},
      Slots{slots} {
    for (const auto &mip : DepthPyramidMips) slots.SetTexture(mip.Slot, *mip.View);
}

RenderTargets::ResourcesT::~ResourcesT() {
    for (const auto &mip : DepthPyramidMips) Slots.Release({SlotType::Image, mip.Slot});
}

RenderTargets::TransmissionResourcesT::TransmissionResourcesT(const mtl::Context &ctx, mtl::Extent2D extent)
    : Image{mtl::CreateTexture2D(ctx, Format::HdrColor, extent, MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead, mtl::MipLevelCount(extent.Width, extent.Height))},
      Mip0View{mtl::CreateMipView(Image, 0)},
      Sampler{mtl::CreateSampler(ctx, MTL::SamplerMinMagFilterLinear, MTL::SamplerMipFilterLinear, MTL::SamplerAddressModeClampToEdge)} {}

RenderTargets::MotionBlurResourcesT::MotionBlurResourcesT(const mtl::Context &ctx, mtl::Extent2D extent, bool fast)
    : OutputImage{mtl::CreateTexture2D(ctx, fast ? Format::HdrColor : MTL::PixelFormatRGBA32Float, extent, MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead)} {
    if (!fast) return;
    const mtl::Extent2D tiles{(extent.Width + 31u) / 32u, (extent.Height + 31u) / 32u};
    VelocityImage = mtl::CreateTexture2D(ctx, Format::HdrColor, extent, MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite);
    TileImage = mtl::CreateTexture2D(ctx, Format::HdrColor, tiles, MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite);
    TileIndirection = mtl::NewBuffer(ctx, 2u * tiles.Width * tiles.Height * sizeof(uint32_t));
}

void RenderTargets::SetExtent(const mtl::Context &ctx, mtl::Extent2D extent, mtl::BindlessSet &slots) {
    Resources = std::make_unique<ResourcesT>(ctx, extent, slots);
    Transmission.reset();
    MotionBlur.reset();
}

bool RenderTargets::EnsureTransmissionResources(const mtl::Context &ctx, mtl::Extent2D extent, bool wanted) {
    if (!wanted) {
        if (!Transmission) return false;
        Transmission.reset();
        return true;
    }
    // Metal rejects zero-sized targets.
    if (extent.Width == 0 || extent.Height == 0) return false;
    if (Transmission && Transmission->Image.Extent == extent) return false;
    Transmission = std::make_unique<TransmissionResourcesT>(ctx, extent);
    return true;
}

bool RenderTargets::EnsureMotionBlurResources(const mtl::Context &ctx, bool fast) {
    if (!Resources || (MotionBlur && bool(MotionBlur->VelocityImage) == fast)) return false; // SetExtent drops it, so an allocated target is always at the color extent.
    MotionBlur = std::make_unique<MotionBlurResourcesT>(ctx, Resources->SceneColorImage.Extent, fast);
    return true;
}

SampledTexture RenderTargets::Nearest(const mtl::Texture *image) const {
    if (!Resources) return {};
    return {image ? **image : *Resources->SceneColorImage, Resources->NearestSampler.get()};
}
SampledTexture RenderTargets::SceneColorSampler() const { return Nearest(nullptr); }
SampledTexture RenderTargets::OverlayColorSampler() const { return Nearest(Resources ? &Resources->OverlayColorImage : nullptr); }
SampledTexture RenderTargets::SceneDepthSampler() const { return Nearest(Resources ? &Resources->VisibilityDepth : nullptr); }
SampledTexture RenderTargets::DepthPyramidSampler() const { return Nearest(Resources ? &Resources->DepthPyramidImage : nullptr); }
SampledTexture RenderTargets::MotionBlurOutputSampler() const { return Nearest(MotionBlur ? &MotionBlur->OutputImage : nullptr); }
SampledTexture RenderTargets::TransmissionSampler() const {
    if (!Resources) return {};
    if (!Transmission) return SceneColorSampler();
    return {*Transmission->Image, Transmission->Sampler.get()};
}
