#pragma once

#include "metal/Image.h"

#include <Metal/MTLBuffer.hpp>
#include <memory>
#include <vector>

namespace mtl {
struct BindlessSet;
} // namespace mtl

struct SampledTexture {
    MTL::Texture *Texture{nullptr};
    MTL::SamplerState *Sampler{nullptr};
    explicit operator bool() const { return Texture != nullptr; }
};

// The viewport-sized attachments and the lazily allocated transmission and motion-blur targets.
struct RenderTargets {
    struct ResourcesT {
        ResourcesT(const mtl::Context &, mtl::Extent2D, mtl::BindlessSet &);
        ~ResourcesT();

        struct PyramidMip {
            mtl::Texture View;
            uint32_t Slot;
            mtl::Extent2D Extent;
        };

        // Visibility IDs and their raster depth stay immutable until visibility consumers finish.
        // Scene-linear color and display-referred overlays stay separate until compositing.
        mtl::Texture VisibilityDepth, ScratchDepth, VisibilityImage, SilhouetteImage, SceneColorImage, OverlayColorImage, FinalColorImage;
        mtl::Texture DepthPyramidImage;
        std::vector<PyramidMip> DepthPyramidMips;
        NS::SharedPtr<MTL::SamplerState> NearestSampler;
        mtl::BindlessSet &Slots;
        bool DepthPyramidValid{false};
    };

    // Lazily allocated, unexposed radiance sampled by real transmission.
    struct TransmissionResourcesT {
        TransmissionResourcesT(const mtl::Context &, mtl::Extent2D);

        mtl::Texture Image;
        mtl::Texture Mip0View;
        NS::SharedPtr<MTL::SamplerState> Sampler;
    };

    // Lazily allocated blur output and, for fast reconstruction, motion tiles.
    struct MotionBlurResourcesT {
        MotionBlurResourcesT(const mtl::Context &, mtl::Extent2D, bool fast);

        mtl::Texture OutputImage, VelocityImage, TileImage;
        NS::SharedPtr<MTL::Buffer> TileIndirection;
    };

    void SetExtent(const mtl::Context &, mtl::Extent2D, mtl::BindlessSet &);
    // Returns whether the allocation changed.
    bool EnsureTransmissionResources(const mtl::Context &, mtl::Extent2D, bool wanted);
    bool EnsureMotionBlurResources(const mtl::Context &, bool fast);

    // Null lazy targets fall back to scene color to keep bindings valid.
    SampledTexture Nearest(const mtl::Texture *) const;
    SampledTexture SceneColorSampler() const;
    SampledTexture OverlayColorSampler() const;
    SampledTexture TransmissionSampler() const;
    SampledTexture MotionBlurOutputSampler() const;
    SampledTexture SceneDepthSampler() const;
    SampledTexture DepthPyramidSampler() const;

    mtl::Extent2D BuiltColorExtent() const { return Resources ? Resources->SceneColorImage.Extent : mtl::Extent2D{}; }

    std::unique_ptr<ResourcesT> Resources;
    std::unique_ptr<TransmissionResourcesT> Transmission;
    std::unique_ptr<MotionBlurResourcesT> MotionBlur;
};
