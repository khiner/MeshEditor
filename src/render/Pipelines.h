#pragma once

#include "gpu/Element.h"
#include "metal/Command.h"
#include "metal/Program.h"
#include "metal/Shader.h"
#include "gpu/PbrFeature.h"

#include "render/RenderSchema.h"
#include "state/Entity.h"

namespace Format = mtl::Format;

// The PBR pipelines specialized to one viewport feature mask.
struct PbrVariant {
    PbrFeatureMask Mask;
    std::optional<mtl::ComputePipeline> RayDirect, RayDirectFiltered, PreparePrimary, GlassWeights, Special, Generate, Shade, ShadeOpaque, OpaqueLayer, Reconstruct;
    // With the patterns feature, the hit kernels that shade the samples whose first surface the hit driver evaluated.
    std::optional<mtl::ComputePipeline> ShadeResume, ShadeOpaqueResume;
    mtl::RenderPipeline Transparent, Visibility;
    std::optional<mtl::RenderPipeline> Prepass; // Present with the transmission feature.
    // With the patterns feature, the transparent pipeline of the patterned materials' fragments, and the pattern list pass that computes their inputs before it.
    std::optional<mtl::RenderPipeline> TransparentPatterns;
    mtl::RenderDriver *PatternList{};
};

// The material preview's lit pipeline for the features, writing radiance to one HDR color attachment.
mtl::RenderPipeline MakePbrPreviewPipeline(mtl::LibraryCache &, PbrFeatureMask);

// Specializes PBR pipelines to the features each viewport shades with and to the output attachments.
// Variants stay compiled, so a frame in flight keeps the pipelines it bound.
struct PbrCompiler {
    PbrCompiler(mtl::LibraryCache &, mtl::PassFormats scene);

    // The variant for the mask, compiled on first use.
    // A mask with the patterns feature links the programs of `programs` into its pattern list pass.
    const PbrVariant &Variant(PbrFeatureMask, mtl::ProgramRegistry *programs);

private:
    mtl::LibraryCache &Libraries;
    mtl::PassFormats SceneFormats;
    std::vector<PbrVariant> Variants;
};

struct MainPipeline {
    MainPipeline(mtl::LibraryCache &);

    mtl::RenderPipeline Background, TransmissionComposite, MotionBlurResolve;
    mtl::RenderPipeline Grid, SilhouetteEdgeColor;
    mtl::RenderPipeline PrepassBackground;
    mtl::RenderPipeline ViewportComposite;
    mtl::RenderPipeline MotionBlurAccumulate, MotionBlurGather;
    mtl::ComputePipeline FastGiDepth, FastGiDepthReduce, FastGiSetup, FastGiReduce, FastGiScan, FastGiDenoise, FastGiResolve;
    mtl::ComputePipeline RayCameraMedia, RayDeformation, RayTemporal, LightCull, LightPrepare, RayShadowBounds, RayBilateral, RayAccumulate, RayFilmReproject, RayFilmReprojectLayer, RayEnvironmentRows, RayEnvironmentMarginal, RayClassify, RayDispatch, RayReset;
    mtl::ComputePipeline MotionBlurTilesFlatten, MotionBlurTilesDilate;
    mtl::RenderPipeline WorkspaceVisibility, WorkspaceTransparent;
    mtl::RenderPipeline TransparencyInit, TransparencyResolve;
    mtl::RenderPipeline MeshletVisibilityOpaque, MeshletVisibilityCoverage, MeshletVisibilityLayer, MeshletVisibilityMotion;
    mtl::RenderPipeline MeshletEditEdges, MeshletEditSmoothEdges;
    mtl::RenderPipeline VertexBlockPoints;
    mtl::RenderPipeline FaceNormalMesh, VertexNormalMesh, OverlayJobLines;
    mtl::RenderPipeline BoneFillMesh, BoneWireMesh, BoneSphereFillMesh, BoneSphereWireMesh;
    mtl::RenderPipeline WireResolve;
    mtl::RenderPipeline ObjectOrigins;

    PbrCompiler Compiler;
};

struct SelectionFragmentPipeline {
    SelectionFragmentPipeline(mtl::LibraryCache &);
    const mtl::RenderPipeline &ElementRaster(Element, bool bitset_box, bool xray) const;

    using ElementVariants = std::array<mtl::RenderPipeline, 4>;
    ElementVariants MeshletFaces, VertexBlocks, MeshletEdges;
    mtl::RenderPipeline MeshletFaceXRayPointsBitsetBox, MeshletEdgeXRayPointsBitsetBox;
    mtl::RenderPipeline ObjectPick, OverlayJobLines, BoneSolid, BoneSphere;
};

namespace ThreadgroupSize {
inline const MTL::Size Linear256{256, 1, 1};
inline const MTL::Size Linear32{32, 1, 1};
inline const MTL::Size Tile16{16, 16, 1};
inline const MTL::Size Tile8{8, 8, 1};
} // namespace ThreadgroupSize

namespace ThreadgroupMemory {
// One aligned float3 per SIMD group's partial min or max.
inline constexpr uint32_t BoundsFoldVector{8 * sizeof(float) * 4};
inline constexpr uint32_t DepthPyramidTile{32 * 32 * sizeof(float)};
} // namespace ThreadgroupMemory

struct Pipelines {
    Pipelines(mtl::LibraryCache &);

    MainPipeline Main;
    mtl::RenderPipeline SilhouetteSeed, Silhouette;
    SelectionFragmentPipeline SelectionFragment;
    mtl::ComputePipeline VisibilityObjectSelection, OutlineOccluderSeed, GeometryEdit, FinalizeElementWork;
    // Materializes current-pose positions before bounds and normal derivation.
    mtl::ComputePipeline PosePrepass;
    mtl::ComputePipeline PosedMeshletBounds;
    // Folds each entry's partial AABBs.
    mtl::ComputePipeline BoundsCombine;
    // Accumulates per-class wire coverage into the screen buffer.
    mtl::ComputePipeline WireRaster;
    // Descends every span tree in lockstep, one count/prefix/emit level at a time.
    mtl::ComputePipeline LodFrontierCount, LodFrontierPrefix, LodFrontierEmit;
    mtl::ComputePipeline MeshletCullBlockCount, MeshletCullPrefix, MeshletCullEmit, SilhouetteCullSize;
    mtl::ComputePipeline OverlayJobBlockCount, OverlayJobPrefix, OverlayJobEmit;
    mtl::ComputePipeline DepthPyramidReduce;
    // The environment prefilter passes run once per loaded environment and bind their textures directly.
    mtl::ComputePipeline EquirectToCubemap, DiffuseIrradiance, SpecularPrefilter;
    // Fills a texture's mip chain from its first level.
    mtl::ComputePipeline MipDownsample;
};

// Returns the pipeline set, compiling every pipeline on first use.
Pipelines &GetPipelines(state::Registry &);
