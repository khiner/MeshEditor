#pragma once

#include "gpu/Element.h"
#include "metal/Image.h"
#include "metal/Shader.h"
#include "render/IblPrefilterPipelines.h"
#include "render/MeshConnectivityPipelines.h"
#include "render/PbrFeature.h"
#include "render/ShaderPipelineType.h"
#include "render/VertexAdjacencyPipelines.h"
#include "render/VertexWeldPipelines.h"

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
    PbrCompiler(mtl::PassFormats scene, mtl::PassFormats scene_velocity);

    enum class Variant {
        Opaque,
        Blend,
        OpaqueVelocity,
        BlendVelocity,
        OpaquePrepass
    };
    static constexpr size_t VariantCount{5};

    bool CompilePipelines(mtl::LibraryCache &, PbrFeatureMask, bool non_triangle_topology);
    bool CompileTopologyPipelines(mtl::LibraryCache &libraries, bool non_triangle_topology) {
        return CompilePipelines(libraries, Mask, non_triangle_topology);
    }
    void BindMeshlets(MTL::RenderCommandEncoder *, Variant) const;
    void BindVisibility(MTL::RenderCommandEncoder *, Variant) const;
    bool HasFeature(PbrFeature f) const { return ::HasFeature(Mask, f); }
    void RecompileModules(mtl::LibraryCache &);

private:
    std::unique_ptr<mtl::MeshRenderPipeline> CreateMeshletPipeline(mtl::LibraryCache &, PbrFeatureMask, bool prepass, Variant, bool) const;
    std::unique_ptr<mtl::RenderPipeline> CreateVisibilityPipeline(mtl::LibraryCache &, PbrFeatureMask, bool prepass, Variant, bool) const;

    mtl::PassFormats SceneFormats, VelocityFormats;
    PbrFeatureMask Mask{0};
    bool NonTriangleTopology{false};
    std::array<std::unique_ptr<mtl::MeshRenderPipeline>, VariantCount> MeshletVariants;
    std::array<std::unique_ptr<mtl::RenderPipeline>, VariantCount> VisibilityVariants;
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

        // Scene-linear color and display-referred overlays stay separate until compositing.
        mtl::Texture DepthImage, VisibilityImage, SceneColorImage, OverlayColorImage, LineDataImage, FinalColorImage;
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

    // Lazily allocated motion vectors, tile reduction, gather, and accumulation targets.
    struct MotionBlurResourcesT {
        MotionBlurResourcesT(const mtl::Context &, mtl::Extent2D);

        mtl::Texture AccumImage, VelocityImage, TileImage, GatherImage;
    };

    void SetExtent(const mtl::Context &, mtl::Extent2D, mtl::BindlessSet &);
    // Returns whether the allocation changed.
    bool EnsureTransmissionResources(const mtl::Context &, mtl::Extent2D, bool wanted);
    bool EnsureMotionBlurResources(const mtl::Context &);

    // Null lazy targets fall back to scene color to keep bindings valid.
    SampledTexture Nearest(const mtl::Texture *) const;
    SampledTexture SceneColorSampler() const;
    SampledTexture OverlayColorSampler() const;
    SampledTexture TransmissionSampler() const;
    SampledTexture MotionBlurAccumSampler() const;
    SampledTexture VelocitySampler() const;
    SampledTexture SceneDepthSampler() const;
    SampledTexture DepthPyramidSampler() const;
    MTL::Texture *MotionBlurTileImage() const;
    SampledTexture MotionBlurGatherSampler() const;

    PipelineRenderer SceneRenderer, OverlayRenderer;
    PipelineRenderer SceneVelocityRenderer;
    mtl::RenderPipeline PrepassBackground;
    mtl::PassFormats CompositeFormats;
    mtl::RenderPipeline ViewportComposite;
    mtl::PassFormats MotionBlurAccumFormats;
    mtl::RenderPipeline MotionBlurAccumulate;
    mtl::PassFormats MotionBlurGatherFormats;
    mtl::RenderPipeline MotionBlurGather;
    mtl::RenderPipeline WorkspaceVisibility;
    mtl::MeshRenderPipeline MeshletVisibilityOpaque, MeshletVisibilityCoverage;
    mtl::MeshRenderPipeline MeshletEditEdges, MeshletEditSmoothEdges;
    mtl::MeshRenderPipeline MeshletEditPoint;
    mtl::MeshRenderPipeline FaceNormalMesh, VertexNormalMesh, OverlayJobLines;
    mtl::MeshRenderPipeline BoneFillMesh, BoneWireMesh, BoneSphereFillMesh, BoneSphereWireMesh;
    // Wireframe lines rasterize in compute, and this resolves their coverage into the overlay layer.
    mtl::RenderPipeline WireResolve;
    std::unique_ptr<ResourcesT> Resources;
    std::unique_ptr<TransmissionResourcesT> Transmission;
    std::unique_ptr<MotionBlurResourcesT> MotionBlur;

    PbrCompiler Compiler;
};

struct SilhouettePipeline {
    SilhouettePipeline(mtl::LibraryCache &);

    struct ResourcesT {
        ResourcesT(const mtl::Context &, mtl::Extent2D);

        mtl::Texture DepthImage, OffscreenImage;
        NS::SharedPtr<MTL::SamplerState> ImageSampler;
    };

    void SetExtent(const mtl::Context &, mtl::Extent2D);

    mtl::RenderPipeline Visibility;
    std::unique_ptr<ResourcesT> Resources;
};

struct SilhouetteEdgePipeline {
    SilhouetteEdgePipeline(mtl::LibraryCache &);

    struct ResourcesT {
        ResourcesT(const mtl::Context &, mtl::Extent2D);

        mtl::Texture DepthImage, OffscreenImage;
        NS::SharedPtr<MTL::SamplerState> ImageSampler;
    };

    void SetExtent(const mtl::Context &, mtl::Extent2D);

    PipelineRenderer Renderer;
    std::unique_ptr<ResourcesT> Resources;
};

struct SelectionFragmentPipeline {
    SelectionFragmentPipeline(mtl::LibraryCache &);
    const mtl::MeshRenderPipeline &ElementRaster(Element, bool bitset_box, bool xray) const;

    using ElementVariants = std::array<mtl::MeshRenderPipeline, 4>;
    ElementVariants MeshletFaces, MeshletVertices, MeshletEdges;
    mtl::MeshRenderPipeline MeshletFaceXRayPointsBitsetBox, MeshletEdgeXRayPointsBitsetBox;
    mtl::MeshRenderPipeline OverlayJobLines, BoneSphere;
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
// Flatten broadcasts a payload and two motion vectors.
inline constexpr uint32_t MotionBlurPayload{16}; // Two uints, padded to Metal's 16-byte threadgroup length granule.
inline constexpr uint32_t MotionBlurMaxMotion{2 * 2 * sizeof(float)};
// One sum per simd group in a 256-thread scan, plus the threadgroup total, padded to Metal's 16-byte granule.
inline constexpr uint32_t BlockScan{48};
} // namespace ThreadgroupMemory

struct Pipelines {
    Pipelines(const mtl::Context &, mtl::LibraryCache &);

    const mtl::Context &Ctx;
    mtl::LibraryCache &Libraries;
    MainPipeline Main;
    SilhouettePipeline Silhouette;
    SilhouetteEdgePipeline SilhouetteEdge;
    SelectionFragmentPipeline SelectionFragment;
    mtl::ComputePipeline VisibilityObjectSelection, PrepareEditSelection, FillEditSelectionList, ResetEditSelectionSummary, DeriveEditSelection, EditSharpness, CommitPosedGeometry;
    // Materializes current-pose positions before bounds and normal derivation.
    mtl::ComputePipeline PosePrepass;
    mtl::ComputePipeline PosedMeshletBounds;
    // Fan-sums face areas, then gathers corner-angle-weighted vertex and seam normals.
    mtl::ComputePipeline VertexNormalDerive;
    // Reduce 256-vertex tiles, then fold each entry's partial AABBs.
    mtl::ComputePipeline BoundsReduce;
    mtl::ComputePipeline BoundsCombine;
    // Walks each wire draw's edge list, accumulating per-class coverage into the screen buffer.
    mtl::ComputePipeline WireRaster;
    // Descends every span tree in lockstep, one count/prefix/emit level at a time.
    mtl::ComputePipeline LodFrontierCount, LodFrontierPrefix, LodFrontierEmit;
    mtl::ComputePipeline MeshletCullBlockCount, MeshletCullPrefix, MeshletCullEmit;
    mtl::ComputePipeline MeshletPhase2Cull, MeshletPhase2RangeCull, MeshletPhase2Prefix;
    mtl::ComputePipeline OverlayJobBlockCount, OverlayJobPrefix, OverlayJobEmit;
    mtl::ComputePipeline DepthPyramidReduce;
    // Marks each tile intersected by the largest motion from every source tile.
    mtl::ComputePipeline MotionBlurTilesFlatten, MotionBlurTilesDilate;
    IblPrefilterPipelines IblPrefilter;
    // Mesh creation runs these three.
    VertexAdjacencyPipelines VertexAdjacency;
    VertexWeldPipelines VertexWeld;
    MeshConnectivityPipelines MeshConnectivity;

    void SetExtent(mtl::Extent2D, mtl::BindlessSet &);
    void CompileShaders();

    mtl::Extent2D BuiltColorExtent() const { return Main.Resources ? Main.Resources->SceneColorImage.Extent : mtl::Extent2D{}; }
};
