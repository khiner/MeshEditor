#pragma once

#include "gpu/Element.h"
#include "metal/Shader.h"
#include "render/PbrFeature.h"

#include "state/Entity.h"
#include <array>
#include <optional>

namespace Format = mtl::Format;

// Specializes PBR pipelines to the scene's active features and output attachments.
struct PbrCompiler {
    PbrCompiler(mtl::LibraryCache &, mtl::PassFormats scene);

    bool CompilePipelines(PbrFeatureMask, bool non_triangle_topology);
    bool CompileTopologyPipelines(bool non_triangle_topology) { return CompilePipelines(Mask, non_triangle_topology); }
    void BindMeshlets(MTL::RenderCommandEncoder *) const;
    void BindVisibility(MTL::RenderCommandEncoder *, bool prepass = false) const;
    bool HasFeature(PbrFeature f) const { return ::HasFeature(Mask, f); }

private:
    mtl::LibraryCache &Libraries;
    mtl::PassFormats SceneFormats;
    PbrFeatureMask Mask{0};
    bool NonTriangleTopology{false};
    std::optional<mtl::RenderPipeline> Transparent, Visibility, Prepass;
};

struct MainPipeline {
    MainPipeline(mtl::LibraryCache &);

    mtl::RenderPipeline Background, TransmissionComposite, MotionBlurResolve;
    mtl::RenderPipeline Grid, SilhouetteEdgeColor;
    mtl::RenderPipeline PrepassBackground;
    mtl::RenderPipeline ViewportComposite;
    mtl::RenderPipeline MotionBlurAccumulate, MotionBlurGather;
    mtl::ComputePipeline MotionBlurTilesFlatten, MotionBlurTilesDilate;
    mtl::RenderPipeline WorkspaceVisibility, WorkspaceTransparent;
    mtl::RenderPipeline TransparencyInit, TransparencyResolve;
    mtl::RenderPipeline MeshletVisibilityOpaque, MeshletVisibilityCoverage;
    mtl::RenderPipeline MeshletEditEdges, MeshletEditSmoothEdges;
    mtl::RenderPipeline MeshletEditPoint;
    mtl::RenderPipeline FaceNormalMesh, VertexNormalMesh, OverlayJobLines;
    mtl::RenderPipeline BoneFillMesh, BoneWireMesh, BoneSphereFillMesh, BoneSphereWireMesh;
    mtl::RenderPipeline WireResolve;

    PbrCompiler Compiler;
};

struct SelectionFragmentPipeline {
    SelectionFragmentPipeline(mtl::LibraryCache &);
    const mtl::RenderPipeline &ElementRaster(Element, bool bitset_box, bool xray) const;

    using ElementVariants = std::array<mtl::RenderPipeline, 4>;
    ElementVariants MeshletFaces, MeshletVertices, MeshletEdges;
    mtl::RenderPipeline MeshletFaceXRayPointsBitsetBox, MeshletEdgeXRayPointsBitsetBox;
    mtl::RenderPipeline ObjectPick, OverlayJobLines, BoneSolid, BoneSphere;
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
    // The environment prefilter passes run once per loaded environment and bind their textures directly.
    mtl::ComputePipeline EquirectToCubemap, DiffuseIrradiance, SpecularPrefilter;
};

// Returns the pipeline set, compiling every pipeline on first use.
Pipelines &GetPipelines(state::Scene &);
