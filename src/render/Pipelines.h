#pragma once

#include "gpu/Element.h"
#include "metal/Image.h"
#include "metal/Shader.h"
#include "render/IblPrefilterPipelines.h"
#include "render/PbrFeature.h"
#include "render/ShaderPipelineType.h"

#include <Metal/MTLBuffer.hpp>
#include <array>
#include <memory>
#include <unordered_map>

namespace mtl {
struct BindlessSet;
} // namespace mtl

using SPT = ShaderPipelineType;

namespace Format = mtl::Format;

// Pipelines sharing attachment formats, which Metal bakes into pipeline state.
struct PipelineRenderer {
    mtl::PassFormats Formats;
    std::unordered_map<SPT, mtl::RenderPipeline> Pipelines;

    void CompileShaders(mtl::LibraryCache &);
    const mtl::RenderPipeline &Bind(MTL::RenderCommandEncoder *, SPT) const;
};

struct SampledTexture {
    MTL::Texture *Texture{nullptr};
    MTL::SamplerState *Sampler{nullptr};
    explicit operator bool() const { return Texture != nullptr; }
};

// Specializes PBR pipelines to the scene's active features and output attachments.
struct PbrCompiler {
    PbrCompiler(mtl::PassFormats scene);

    bool CompilePipelines(mtl::LibraryCache &, PbrFeatureMask, bool non_triangle_topology);
    bool CompileTopologyPipelines(mtl::LibraryCache &libraries, bool non_triangle_topology) {
        return CompilePipelines(libraries, Mask, non_triangle_topology);
    }
    void BindMeshlets(MTL::RenderCommandEncoder *) const;
    void BindVisibility(MTL::RenderCommandEncoder *, bool prepass = false) const;
    bool HasFeature(PbrFeature f) const { return ::HasFeature(Mask, f); }
    void RecompileModules(mtl::LibraryCache &);

private:
    mtl::PassFormats SceneFormats;
    PbrFeatureMask Mask{0};
    bool NonTriangleTopology{false};
    std::unique_ptr<mtl::MeshRenderPipeline> Transparent;
    std::unique_ptr<mtl::RenderPipeline> Visibility, Prepass;
};

struct MainPipeline {
    MainPipeline(mtl::LibraryCache &);

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

    PipelineRenderer SceneRenderer, OverlayRenderer;
    mtl::RenderPipeline PrepassBackground;
    mtl::RenderPipeline ViewportComposite;
    mtl::RenderPipeline MotionBlurAccumulate, MotionBlurGather;
    mtl::ComputePipeline MotionBlurTilesFlatten, MotionBlurTilesDilate;
    mtl::RenderPipeline WorkspaceVisibility;
    mtl::RenderPipeline TransparencyInit, TransparencyResolve;
    mtl::MeshRenderPipeline MeshletVisibilityOpaque, MeshletVisibilityCoverage;
    mtl::MeshRenderPipeline MeshletEditEdges, MeshletEditSmoothEdges;
    mtl::MeshRenderPipeline MeshletEditPoint;
    mtl::MeshRenderPipeline FaceNormalMesh, VertexNormalMesh, OverlayJobLines;
    mtl::MeshRenderPipeline BoneFillMesh, BoneWireMesh, BoneSphereFillMesh, BoneSphereWireMesh;
    mtl::RenderPipeline WireResolve;
    std::unique_ptr<ResourcesT> Resources;
    std::unique_ptr<TransmissionResourcesT> Transmission;
    std::unique_ptr<MotionBlurResourcesT> MotionBlur;

    PbrCompiler Compiler;
};

struct SelectionFragmentPipeline {
    SelectionFragmentPipeline(mtl::LibraryCache &);
    const mtl::MeshRenderPipeline &ElementRaster(Element, bool bitset_box, bool xray) const;

    using ElementVariants = std::array<mtl::MeshRenderPipeline, 4>;
    ElementVariants MeshletFaces, MeshletVertices, MeshletEdges;
    mtl::MeshRenderPipeline MeshletFaceXRayPointsBitsetBox, MeshletEdgeXRayPointsBitsetBox;
    mtl::MeshRenderPipeline ObjectPick, OverlayJobLines, BoneSphere;
};

namespace ThreadgroupSize {
inline const MTL::Size Linear256{256, 1, 1};
inline const MTL::Size Linear64{64, 1, 1};
inline const MTL::Size Tile16{16, 16, 1};
inline const MTL::Size Tile8{8, 8, 1};
} // namespace ThreadgroupSize

namespace ThreadgroupMemory {
// One min and max float4 per bounds lane.
inline constexpr uint32_t BoundsFoldVector{256 * sizeof(float) * 4};
inline constexpr uint32_t MeshletBoundsFoldVector{64 * sizeof(float) * 4};
inline constexpr uint32_t DepthPyramidTile{32 * 32 * sizeof(float)};
} // namespace ThreadgroupMemory

struct Pipelines {
    Pipelines(mtl::LibraryCache &);

    mtl::LibraryCache &Libraries;
    MainPipeline Main;
    mtl::RenderPipeline Silhouette;
    SelectionFragmentPipeline SelectionFragment;
    mtl::ComputePipeline VisibilityObjectSelection, PrepareEditSelection, FillEditSelectionList, ResetEditSelectionSummary, DeriveEditSelection, SumEditSelectionPosition, EditSharpness, CommitPosedGeometry, GeometryWorkArgs;
    // Materializes current-pose positions before bounds and normal derivation.
    mtl::ComputePipeline PosePrepass;
    mtl::ComputePipeline PosedMeshletBounds;
    // Fan-sums face areas, then gathers corner-angle-weighted vertex and seam normals.
    mtl::ComputePipeline VertexNormalDerive;
    // Reduce 256-vertex tiles, then fold each entry's partial AABBs.
    mtl::ComputePipeline BoundsReduce;
    mtl::ComputePipeline BoundsCombine;
    mtl::ComputePipeline BoundsTree;
    // Accumulates per-class wire coverage into the screen buffer.
    mtl::ComputePipeline WireRaster;
    // Descends every span tree in lockstep, one count/prefix/emit level at a time.
    mtl::ComputePipeline LodFrontierCount, LodFrontierPrefix, LodFrontierEmit;
    mtl::ComputePipeline MeshletCullBlockCount, MeshletCullPrefix, MeshletCullEmit;
    mtl::ComputePipeline OverlayJobBlockCount, OverlayJobPrefix, OverlayJobEmit;
    mtl::ComputePipeline DepthPyramidReduce;
    IblPrefilterPipelines IblPrefilter;
    // Mesh creation runs these three.

    void CompileShaders();

    mtl::Extent2D BuiltColorExtent() const { return Main.Resources ? Main.Resources->SceneColorImage.Extent : mtl::Extent2D{}; }
};
