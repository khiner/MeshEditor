#pragma once

#include "metal/Bindless.h"
#include "metal/Buffer.h"
#include "metal/Image.h"
#include "metal/Mipmaps.h"
#include "render/RenderSchema.h"
#include "project/store/Hash.h"

struct SampledTexture {
    MTL::Texture *Texture{nullptr};
    MTL::SamplerState *Sampler{nullptr};
    explicit operator bool() const { return Texture != nullptr; }
};

// Bindless sampler slots the render passes read the viewport targets through, fixed for the viewport's lifetime.
struct RenderSamplerSlots {
    explicit RenderSamplerSlots(mtl::BindlessSet &slots)
        : Owner(slots),
          Silhouette(Owner.Allocate(SlotType::Sampler)), SceneColor(Owner.Allocate(SlotType::Sampler)), OverlayColor(Owner.Allocate(SlotType::Sampler)),
          Transmission(Owner.Allocate(SlotType::Sampler)), MotionBlurOutput(Owner.Allocate(SlotType::Sampler)), Velocity(Owner.Allocate(SlotType::Sampler)),
          SceneDepth(Owner.Allocate(SlotType::Sampler)), DepthPyramid(Owner.Allocate(SlotType::Sampler)),
          OutlineOccluderPyramid(Owner.Allocate(SlotType::Sampler)) {}

    mtl::SlotOwner Owner;
    uint32_t Silhouette, SceneColor, OverlayColor, Transmission, MotionBlurOutput, Velocity, SceneDepth, DepthPyramid, OutlineOccluderPyramid;
};

// The screen-space targets and the lazily allocated transmission and motion-blur targets.
struct RenderTargets {
    struct ScreenT {
        ScreenT(const mtl::Context &, mtl::Extent2D, mtl::BindlessSet &);
        ~ScreenT();

        struct PyramidMip {
            mtl::Texture View;
            uint32_t Slot;
            mtl::Extent2D Extent;
        };
        struct Pyramid {
            mtl::Texture Image;
            std::vector<PyramidMip> Mips;
        };

        // Visibility IDs and their raster depth stay immutable until visibility consumers finish.
        // Scene-linear color and display-referred overlays stay separate until compositing.
        mtl::Texture VisibilityDepth, ScratchDepth, VisibilityImage, SilhouetteImage, SceneColorImage, OverlayColorImage, FinalColorImage;
        Pyramid DepthPyramid; // Farthest visibility depth, for the next frame's occlusion cull.
        Pyramid OutlineOccluderPyramid; // Nearest depth of unoutlined surfaces, for the silhouette cull.
        NS::SharedPtr<MTL::SamplerState> NearestSampler;
        mtl::BindlessSet &Slots;
        bool DepthPyramidValid{false};
        // The view-projection of the cull whose visibility depth built the depth pyramid.
        mat4 DepthPyramidViewProj{1};
    };

    // Lazily allocated, unexposed radiance sampled by real transmission.
    struct TransmissionResourcesT {
        TransmissionResourcesT(const mtl::Context &, mtl::Extent2D);

        mtl::Texture Image;
        mtl::Texture Mip0View;
        mtl::MipChain Mips;
        NS::SharedPtr<MTL::SamplerState> Sampler;
    };

    // Lazily allocated blur output and, for fast reconstruction, motion tiles.
    struct MotionBlurResourcesT {
        MotionBlurResourcesT(const mtl::Context &, mtl::Extent2D, bool fast);

        mtl::Texture OutputImage, VelocityImage, TileImage;
        mtl::ResidentBuffer TileIndirection;
    };

    struct RayTracingResourcesT {
        RayTracingResourcesT(const mtl::Context &, mtl::Extent2D, uint32_t step, uint32_t lobes, bool glass);
        mtl::Texture Radiance, Guide, HitGuide, Spatial, SpatialInfo, RayData, Incident;
        // Glass shape stores the octahedral normal, roughness and thickness in FP32.
        mtl::Texture ClosureNormal, GlassShape;
        struct FastGiT {
            FastGiT(const mtl::Context &, mtl::Extent2D, uint32_t);
            mtl::Texture Depth, Radiance, Normal, SH, FilteredSH;
            std::vector<mtl::Texture> DepthMips, RadianceMips, NormalMips;
        };
        std::unique_ptr<FastGiT> FastGI;
        struct VisibilityLayer {
            VisibilityLayer(const mtl::Context &, mtl::Extent2D);
            mtl::Texture Visibility, Depth;
        };
        struct OpaqueLayerT : VisibilityLayer {
            OpaqueLayerT(const mtl::Context &, mtl::Extent2D);
            mtl::Texture Radiance, Geometry, Normal, Identity;
        };
        std::unique_ptr<OpaqueLayerT> OpaqueLayer;
        std::unique_ptr<VisibilityLayer> FilmLayer;
        std::unique_ptr<mtl::Buffer> LightTiles, PreparedAreaLights;
        // The second closure stores transmission on glass and diffuse on opaque surfaces.
        mtl::Texture Transmission, TransmissionHit, SampleBase, Feedback;
        mtl::Texture ReflectionWeight, TransmissionWeight, TransmissionRayData, TransmissionIncident, GlassDirect;
        struct HistoryT {
            mtl::Texture Radiance, MaterialValidity, Moments, Transmission, TransmissionMoments, Geometry, Normal, Film;
        };
        std::array<HistoryT, 2> History;
        uint32_t Frames{}, SampleIndex{}, JitterIndex{}, HistoryIndex{};
        uint64_t SceneRevision{}, GeometryRevision{};
        mat4 PreviousViewProj{1}, PreviousCameraProj{1};
        float FilmWeight{};
        store::Hash128 ShadingHash{}, LightHash{};
        mtl::ResidentBuffer Work, RayQueue;
    };
    std::unique_ptr<RayTracingResourcesT> RayTracing;

    void SetExtent(const mtl::Context &, mtl::Extent2D, mtl::BindlessSet &);
    // Returns whether the allocation changed.
    bool EnsureTransmissionResources(const mtl::Context &, mtl::Extent2D, bool wanted);
    bool EnsureMotionBlurResources(const mtl::Context &, bool fast);
    RayTracingResourcesT &EnsureRayTracingResources(const mtl::Context &, uint32_t step, uint32_t lobes, bool glass);

    // Null lazy targets fall back to scene color to keep bindings valid.
    SampledTexture Nearest(const mtl::Texture *) const;
    SampledTexture SceneColorSampler() const;
    SampledTexture OverlayColorSampler() const;
    SampledTexture TransmissionSampler() const;
    SampledTexture MotionBlurOutputSampler() const;
    SampledTexture SceneDepthSampler() const;
    SampledTexture DepthPyramidSampler() const;
    SampledTexture OutlineOccluderPyramidSampler() const;

    mtl::Extent2D BuiltColorExtent() const { return Screen ? Screen->SceneColorImage.Extent : mtl::Extent2D{}; }

    std::unique_ptr<ScreenT> Screen;
    std::unique_ptr<TransmissionResourcesT> Transmission;
    std::unique_ptr<MotionBlurResourcesT> MotionBlur;
};
