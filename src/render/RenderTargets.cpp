#include "render/RenderTargets.h"
#include "gpu/MaterialPattern.h"
#include "gpu/RayTracingRecord.h"
#include "gpu/SurfacePattern.h"

#include <algorithm>
#include <bit>

namespace Format = mtl::Format;

namespace {
uint64_t RayQueueBytes(mtl::Extent2D extent, uint32_t lobes) {
    return uint64_t(extent.Width) * extent.Height * lobes * sizeof(RayLightingSample);
}

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
      SceneColorImage{mtl::CreateTexture2D(ctx, Format::HdrColor, extent, MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite)},
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
    const auto tiles = extent.Blocks(32u);
    VelocityImage = mtl::CreateTexture2D(ctx, Format::HdrColor, extent, MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite);
    TileImage = mtl::CreateTexture2D(ctx, Format::HdrColor, tiles, MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite);
    TileIndirection = mtl::ResidentBuffer{ctx, 2u * tiles.Width * tiles.Height * sizeof(uint32_t)};
}

void RenderTargets::SetExtent(const mtl::Context &ctx, mtl::Extent2D extent, mtl::BindlessSet &slots) {
    Screen = std::make_unique<ScreenT>(ctx, extent, slots);
    Transmission.reset();
    MotionBlur.reset();
    RayTracing.reset();
    Patterns.reset();
}

RenderTargets::PatternResourcesT::PatternResourcesT(const mtl::Context &ctx, mtl::Extent2D extent, uint32_t ray_samples)
    : Work{ctx, (uint64_t(extent.Width) * extent.Height + SurfacePatternPixels) * sizeof(uint32_t)}, RaySamples{ray_samples} {
    if (ray_samples == 0u) return;
    Hits.emplace(ctx, (PatternHitEntries + 2u * uint64_t(ray_samples)) * sizeof(uint32_t));
    std::ranges::fill(Hits->Contents().first(PatternHitEntries * sizeof(uint32_t)), std::byte{});
}

RenderTargets::PatternResourcesT *RenderTargets::EnsurePatternResources(const mtl::Context &ctx, mtl::BufferContext &buffers, uint32_t stride, uint32_t ray_samples, bool fragments) {
    if (stride == 0u) {
        Patterns.reset();
        return nullptr;
    }
    const auto extent = Screen->SceneColorImage.Extent; // SetExtent releases Patterns when this changes.
    if (!Patterns || Patterns->RaySamples != ray_samples) {
        auto words = Patterns ? std::move(Patterns->Words) : nullptr;
        const auto held = Patterns ? Patterns->FragmentCapacity : 0u;
        Patterns = std::make_unique<PatternResourcesT>(ctx, extent, ray_samples);
        Patterns->Words = std::move(words);
        Patterns->FragmentCapacity = held;
    }
    const auto pixels = extent.Width * extent.Height;
    if (fragments && !Patterns->FragmentCapacity) Patterns->FragmentCapacity = pixels;
    const auto capacity = fragments ? Patterns->FragmentCapacity : 0u;
    Patterns->Ring = PatternWordsHeader + pixels * stride;
    const auto heads = Patterns->Ring + ray_samples * stride;
    Patterns->Fragments = {.Heads = heads, .Entries = heads + (fragments ? pixels : 0u), .Words = heads + (fragments ? pixels + 2u * capacity : 0u), .Capacity = capacity};
    const auto bytes = (uint64_t(Patterns->Fragments.Words) + uint64_t(capacity) * stride) * sizeof(uint32_t);
    if (!Patterns->Words) Patterns->Words = std::make_unique<mtl::Buffer>(buffers, bytes, SlotType::Buffer, mtl::BufferLifetime::Workspace);
    else Patterns->Words->Reserve(bytes);
    return Patterns.get();
}

bool RenderTargets::FitFragmentLists() {
    if (!Patterns || !std::exchange(Patterns->Listed, false)) return false;
    // The list pass counts every fragment it lists in the header's first word, including those past the lists' end.
    const auto listed = Patterns->Words->GetSpan<uint32_t>({0u, 1u})[0];
    auto &capacity = Patterns->FragmentCapacity;
    Patterns->Overflowed = listed > capacity;
    if (listed > capacity || listed < capacity / 4u) capacity = std::bit_ceil(std::max(listed, 1u));
    return Patterns->Overflowed;
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

RenderTargets::RayTracingResourcesT &RenderTargets::EnsureRayTracingResources(const mtl::Context &ctx, uint32_t step, uint32_t lobes, bool glass, bool coat, bool diffuse_transmission, bool sheen) {
    const auto extent = Screen->SceneColorImage.Extent; // SetExtent releases RayTracing when this changes.
    if (!RayTracing || RayTracing->RayData.Extent != extent.Blocks(step) || RayTracing->RayQueue.Length() != RayQueueBytes(extent, lobes) || bool(RayTracing->GlassShape) != glass ||
        bool(RayTracing->CoatWeight) != coat || bool(RayTracing->DiffuseTransmissionWeight) != diffuse_transmission || bool(RayTracing->SheenWeight) != sheen)
        RayTracing = std::make_unique<RayTracingResourcesT>(ctx, extent, step, lobes, glass, coat, diffuse_transmission, sheen);
    return *RayTracing;
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

RenderTargets::RayTracingResourcesT::RayTracingResourcesT(const mtl::Context &ctx, mtl::Extent2D extent, uint32_t step, uint32_t lobes, bool glass, bool coat, bool diffuse_transmission, bool sheen)
    : Work{ctx, (RayTileEnd(extent.Width, extent.Height) + (lobes > 1u && !glass ? 3u + RayTileCount(extent.Width, extent.Height) : 0u)) * sizeof(uint32_t)},
      RayQueue{ctx, RayQueueBytes(extent, lobes)} {
    constexpr auto usage = MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite;
    // Inactive closures retain a single texel so their argument bindings stay valid.
    const auto texture = [&](MTL::PixelFormat format, bool active = true) {
        return mtl::CreateTexture2D(ctx, format, active ? extent : mtl::Extent2D{1, 1}, usage, 1, MTL::StorageModePrivate);
    };
    const auto samples = extent.Blocks(step);
    const auto ray_samples = [&](bool active = true) {
        return mtl::CreateTexture2D(ctx, MTL::PixelFormatRGBA32Float, active ? samples : mtl::Extent2D{1, 1}, usage);
    };
    Radiance = texture(MTL::PixelFormatRGBA32Float);
    Guide = texture(MTL::PixelFormatRGBA32Float);
    HitGuide = texture(MTL::PixelFormatRGBA32Float);
    ClosureNormal = texture(MTL::PixelFormatRG16Float);
    Spatial = texture(MTL::PixelFormatRGBA16Float);
    SpatialInfo = texture(MTL::PixelFormatRG32Float);
    RayData = ray_samples();
    Incident = ray_samples();
    if (glass) GlassShape = mtl::CreateTexture2D(ctx, MTL::PixelFormatRGBA32Float, extent, usage);
    GlassDirect = texture(MTL::PixelFormatRGBA16Float, glass);
    ReflectionWeight = texture(MTL::PixelFormatRGBA16Float);
    TransmissionWeight = texture(MTL::PixelFormatRGBA16Float);
    TransmissionRayData = ray_samples(lobes > 1u);
    TransmissionIncident = ray_samples(lobes > 1u);
    Transmission = texture(MTL::PixelFormatRGBA32Float, lobes > 1u);
    TransmissionHit = texture(MTL::PixelFormatRGBA32Float, lobes > 1u);
    Feedback = texture(MTL::PixelFormatRGBA16Float);
    if (coat) {
        CoatWeight = texture(MTL::PixelFormatRGBA16Float);
        CoatNormal = texture(MTL::PixelFormatRG16Float);
    }
    if (diffuse_transmission) DiffuseTransmissionWeight = texture(MTL::PixelFormatRGBA16Float);
    if (sheen) SheenWeight = texture(MTL::PixelFormatRGBA16Float);
    for (auto &history : History) {
        history.Film = texture(MTL::PixelFormatRGBA32Float);
        history.Geometry = texture(MTL::PixelFormatRG32Float);
        history.Normal = texture(MTL::PixelFormatRG16Float);
        history.Radiance = texture(MTL::PixelFormatRGBA16Float);
        history.MaterialValidity = texture(MTL::PixelFormatR32Uint);
        history.Moments = texture(MTL::PixelFormatR16Float);
        history.Transmission = texture(MTL::PixelFormatRGBA16Float, lobes > 1u);
        history.TransmissionMoments = texture(MTL::PixelFormatR16Float, lobes > 1u);
    }
}

RenderTargets::RayTracingResourcesT::FastGiT::FastGiT(const mtl::Context &ctx, mtl::Extent2D extent, uint32_t step) {
    const auto usage = MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite;
    const auto samples = extent.Blocks(step);
    Depth = mtl::CreateTexture2D(ctx, MTL::PixelFormatR32Float, extent, usage, mtl::MipLevelCount(extent.Width, extent.Height));
    const auto mips = std::min(4u, mtl::MipLevelCount(samples.Width, samples.Height));
    Radiance = mtl::CreateTexture2D(ctx, MTL::PixelFormatRGBA16Float, samples, usage, mips);
    Normal = mtl::CreateTexture2D(ctx, MTL::PixelFormatRGBA16Float, samples, usage, mips);
    for (uint32_t i = 0; i < Depth.MipLevels; ++i) DepthMips.push_back(mtl::CreateMipView(Depth, i));
    for (uint32_t i = 0; i < mips; ++i) {
        RadianceMips.push_back(mtl::CreateMipView(Radiance, i));
        NormalMips.push_back(mtl::CreateMipView(Normal, i));
    }
    SH = mtl::CreateTexture2DArray(ctx, MTL::PixelFormatRGBA16Float, samples, 4, usage);
    FilteredSH = mtl::CreateTexture2DArray(ctx, MTL::PixelFormatRGBA16Float, samples, 4, usage);
}

RenderTargets::RayTracingResourcesT::VisibilityLayer::VisibilityLayer(const mtl::Context &ctx, mtl::Extent2D extent) {
    const auto usage = MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead;
    Visibility = mtl::CreateTexture2D(ctx, Format::Uint, extent, usage);
    Depth = mtl::CreateTexture2D(ctx, Format::Depth, extent, usage);
}

RenderTargets::RayTracingResourcesT::OpaqueLayerT::OpaqueLayerT(const mtl::Context &ctx, mtl::Extent2D extent)
    : VisibilityLayer(ctx, extent) {
    const auto texture = [&](MTL::PixelFormat format) {
        return mtl::CreateTexture2D(ctx, format, extent, MTL::TextureUsageShaderWrite | MTL::TextureUsageShaderRead);
    };
    Radiance = texture(MTL::PixelFormatRGBA16Float);
    Geometry = texture(MTL::PixelFormatRG32Float);
    Normal = texture(MTL::PixelFormatRG16Float);
    Identity = texture(MTL::PixelFormatRG32Uint);
}
