#include "render/RenderTargets.h"

namespace Format = mtl::Format;

namespace {
RenderTargets::ScreenT::Pyramid MakePyramid(const mtl::Context &ctx, mtl::Extent2D extent, mtl::BindlessSet &slots) {
    const mtl::Extent2D padded{std::bit_ceil((extent.Width + 1) / 2), std::bit_ceil((extent.Height + 1) / 2)};
    auto image = mtl::CreateTexture2D(ctx, Format::Float, padded, MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite, mtl::MipLevelCount(padded.Width, padded.Height));
    std::vector<RenderTargets::ScreenT::PyramidMip> mips;
    mips.reserve(image.MipLevels);
    for (uint32_t mip = 0; mip < image.MipLevels; ++mip) {
        const mtl::Extent2D data_extent{((extent.Width - 1) >> (mip + 1)) + 1, ((extent.Height - 1) >> (mip + 1)) + 1};
        mips.push_back({mtl::CreateMipView(image, mip), slots.Allocate(SlotType::Image), data_extent});
        slots.SetTexture(mips.back().Slot, *mips.back().View);
    }
    return {std::move(image), std::move(mips)};
}
} // namespace

RenderTargets::ScreenT::ScreenT(const mtl::Context &ctx, mtl::Extent2D extent, mtl::BindlessSet &slots)
    // Visibility depth remains paired with its IDs throughout shading and selection.
    : VisibilityDepth{mtl::CreateTexture2D(ctx, Format::Depth, extent, MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead)},
      ScratchDepth{mtl::CreateTexture2D(ctx, Format::Depth, extent, MTL::TextureUsageRenderTarget)},
      VisibilityImage{mtl::CreateTexture2D(ctx, Format::Uint, extent, MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead)},
      SilhouetteImage{mtl::CreateTexture2D(ctx, Format::Float2, extent, MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead)},
      SceneColorImage{mtl::CreateTexture2D(ctx, Format::HdrColor, extent, MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead)},
      OverlayColorImage{mtl::CreateTexture2D(ctx, Format::Color, extent, MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead)},
      FinalColorImage{mtl::CreateTexture2D(ctx, Format::Color, extent, MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead)},
      DepthPyramid{MakePyramid(ctx, extent, slots)},
      OutlineOccluderPyramid{MakePyramid(ctx, extent, slots)},
      NearestSampler{mtl::CreateSampler(ctx, MTL::SamplerMinMagFilterNearest, MTL::SamplerMipFilterNearest, MTL::SamplerAddressModeClampToEdge)},
      Slots{slots} {}

RenderTargets::ScreenT::~ScreenT() {
    for (const auto *pyramid : {&DepthPyramid, &OutlineOccluderPyramid}) {
        for (const auto &mip : pyramid->Mips) Slots.Release({SlotType::Image, mip.Slot});
    }
}

RenderTargets::TransmissionResourcesT::TransmissionResourcesT(const mtl::Context &ctx, mtl::Extent2D extent)
    : Image{mtl::CreateTexture2D(ctx, Format::HdrColor, extent, MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite, mtl::MipLevelCount(extent.Width, extent.Height))},
      Mip0View{mtl::CreateMipView(Image, 0)},
      Mips{mtl::CreateMipChain(Image)},
      Sampler{mtl::CreateSampler(ctx, MTL::SamplerMinMagFilterLinear, MTL::SamplerMipFilterLinear, MTL::SamplerAddressModeClampToEdge)} {}

RenderTargets::MotionBlurResourcesT::MotionBlurResourcesT(const mtl::Context &ctx, mtl::Extent2D extent, bool fast)
    : OutputImage{mtl::CreateTexture2D(ctx, fast ? Format::HdrColor : MTL::PixelFormatRGBA32Float, extent, MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead)} {
    if (!fast) return;
    const mtl::Extent2D tiles{(extent.Width + 31u) / 32u, (extent.Height + 31u) / 32u};
    VelocityImage = mtl::CreateTexture2D(ctx, Format::HdrColor, extent, MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite);
    TileImage = mtl::CreateTexture2D(ctx, Format::HdrColor, tiles, MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite);
    TileIndirection = mtl::ResidentBuffer{ctx, 2u * tiles.Width * tiles.Height * sizeof(uint32_t)};
}

void RenderTargets::SetExtent(const mtl::Context &ctx, mtl::Extent2D extent, mtl::BindlessSet &slots) {
    Screen = std::make_unique<ScreenT>(ctx, extent, slots);
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
    // SetExtent releases MotionBlur when the color extent changes.
    if (!Screen || (MotionBlur && bool(MotionBlur->VelocityImage) == fast)) return false;
    MotionBlur = std::make_unique<MotionBlurResourcesT>(ctx, Screen->SceneColorImage.Extent, fast);
    return true;
}

SampledTexture RenderTargets::Nearest(const mtl::Texture *image) const {
    if (!Screen) return {};
    return {image ? **image : *Screen->SceneColorImage, Screen->NearestSampler.get()};
}
SampledTexture RenderTargets::SceneColorSampler() const { return Nearest(nullptr); }
SampledTexture RenderTargets::OverlayColorSampler() const { return Nearest(Screen ? &Screen->OverlayColorImage : nullptr); }
SampledTexture RenderTargets::SceneDepthSampler() const { return Nearest(Screen ? &Screen->VisibilityDepth : nullptr); }
SampledTexture RenderTargets::DepthPyramidSampler() const { return Nearest(Screen ? &Screen->DepthPyramid.Image : nullptr); }
SampledTexture RenderTargets::OutlineOccluderPyramidSampler() const { return Nearest(Screen ? &Screen->OutlineOccluderPyramid.Image : nullptr); }
SampledTexture RenderTargets::MotionBlurOutputSampler() const { return Nearest(MotionBlur ? &MotionBlur->OutputImage : nullptr); }
SampledTexture RenderTargets::TransmissionSampler() const {
    if (!Screen) return {};
    if (!Transmission) return SceneColorSampler();
    return {*Transmission->Image, Transmission->Sampler.get()};
}
