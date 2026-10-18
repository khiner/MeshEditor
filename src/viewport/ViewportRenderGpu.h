#pragma once

#include "Range.h"
#include "gpu/MeshletRouteMode.h"

#include "state/Entity.h"
#include <span>

namespace MTL {
class CommandBuffer;
class RenderCommandEncoder;
} // namespace MTL

namespace mtl {
struct BindlessSet;
struct Buffer;
struct ComputeChain;
struct PassChain;
} // namespace mtl

struct GpuBuffers;
struct MeshStore;
struct Pipelines;
struct RenderSamplerSlots;
struct RenderTargets;

struct MeshVertexChanges {
    state::Entity Entity;
    std::span<const Range> Ranges;
};
// Recomputes normals, polygon triangles and edit work for changed vertex ranges without mutating the vertices.
// Submits where stale footprints or changed polygon triangles require host-side render repair.
// One refresh records per chain submit.
// History restoration already restores triangle pages and their render ownership.
void RefreshEditedPositions(state::Registry &, mtl::ComputeChain &, std::span<const MeshVertexChanges>, bool retessellate = true);
// Updates captured render triangles against the pending transform and repairs changed clusters.
bool RefreshPreviewTessellation(state::Registry &, state::Entity viewport);
enum class PositionPublication { Commit,
                                 Preview };
// Publishes submitted edits, or a cached preview using its prepared candidate footprint without a readback.
// Preview preserves edit work and updates selection and bounds; commit also invalidates LODs.
// Records the remaining GPU updates on the chain and returns the meshes being published.
std::vector<state::Entity> PublishEditedPositions(state::Registry &, mtl::ComputeChain &, std::span<const state::Entity>, PositionPublication = PositionPublication::Commit);

// A pixel rectangle on a render target.
struct PixelRect {
    uvec2 Origin{}, Extent{};
};

struct MeshletCullConfig {
    MeshletRouteMode Mode{MeshletRouteMode::Single};
    uint32_t RequiredInstanceFlags{0};
    uint32_t RouteMask{0x1ffu};
    uint32_t UboOffset{0};
    uint32_t PyramidSamplerSlot{InvalidSlot};
    bool ExactEditGeometry{false};
    float MinEditOverlayDiameterPixels{0.0f};
    bool EditOverlayHasSharpEdges{false};
    bool EditOutput{false};
};

void RecordMeshletCull(mtl::PassChain &, const mtl::BindlessSet &, const Pipelines &, GpuBuffers &, MeshletCullConfig);
void RecordOverlayJobCull(
    mtl::PassChain &, const mtl::BindlessSet &, const Pipelines &, GpuBuffers &,
    bool extras_only = false, uint32_t ubo_offset = 0
);
void DrawOverlayJobs(MTL::RenderCommandEncoder *, const GpuBuffers &, const MeshStore &);
// Rasterize the current meshlet routes into the visibility/depth pair shared by shading and selection.
void RecordMeshletVisibilityPass(
    mtl::PassChain &, const mtl::BindlessSet &, const Pipelines &, const RenderTargets &, GpuBuffers &,
    bool transmission, uint32_t ubo_offset, std::optional<PixelRect> scissor
);
void RecordSilhouetteDepthPass(mtl::PassChain &, const mtl::BindlessSet &, const Pipelines &, const RenderTargets &, const RenderSamplerSlots &, GpuBuffers &, uint32_t ubo_offset = 0);
void DrawMeshlets(
    MTL::RenderCommandEncoder *, const GpuBuffers &, uint32_t route,
    uint32_t required_instance_flags = 0, uint32_t mesh_threads = 160u,
    uint32_t edit_edge_corner = 0u
);
// Draws the instance record's live vertices of the mesh with the bound pipeline, per canonical vertex block of the mesh.
// A block culls against its static or posed bounds, the depth pyramid at `pyramid_slot`, and a minimum projected diameter.
// Sound points draw only the selected vertices.
void DrawVertexBlocks(
    MTL::RenderCommandEncoder *, const state::Registry &, state::Entity mesh_entity, uint32_t instance, bool sound_points = false,
    uint32_t pyramid_slot = InvalidSlot, float min_diameter_pixels = 0.0f
);

// Which parts of a frame one recording covers.
enum class RenderPhase {
    Prepare,
    Full,
    BlurFast,
    BlurAccumulateFirst,
    BlurAccumulate,
    BlurResolve,
};

constexpr bool IsBlurAccumulate(RenderPhase p) { return p == RenderPhase::BlurAccumulateFirst || p == RenderPhase::BlurAccumulate; }

// Records the phase, whose prelude recomputes the bounds of the meshes MeshBoundsDirty marks, and consumes the marks.
void RecordRenderCommandBuffer(state::Registry &, state::Entity viewport, MTL::CommandBuffer *, RenderPhase = RenderPhase::Full);

// Records every blur step and the resolve into one command buffer using one view-UBO instance per step.
void RecordBlurStepsCommandBuffer(state::Registry &, state::Entity viewport, MTL::CommandBuffer *, std::span<const uint32_t> sample_weights);

// Resolves whether the listed mesh entities retain authored shading normals under morphing.
// The CPU resolves targets with authored normal deltas.
// Position-only targets derive their full-weight poses on the chain, which submits once for them.
// The derived pose tests whether derivation moves the normals authored shading would pin.
// Call after the base derive has run, since the pin test compares against the base normal stores.
void UpdateAuthoredMorphShading(state::Registry &, mtl::ComputeChain &, std::span<const state::Entity> mesh_entities);

// Encodes the authored corner normals of each mesh entity holding them, resolves its authored-morph-shading gate, and drops the stored normals.
// Call once its base normals derived and the host can read them.
void FinalizeAuthoredShading(state::Registry &, mtl::ComputeChain &);

// Records the evaluation of the final pending edit into canonical positions and affected normals on the chain.
// Returns the committed meshes, whose changed vertices PublishEditedPositions reads once the chain submits.
std::vector<state::Entity> CommitPosedGeometry(state::Registry &, mtl::ComputeChain &, state::Entity viewport, std::span<const state::Entity> mesh_entities);
// Removes the mesh's edit work, if any, and rederives its display fields, and its layout when the work required a pose, at the end of the settle pass.
void ReleaseMeshEditWork(state::Registry &, state::Entity mesh_entity);
// The MeshEditWork destroy handler, which releases the work's geometry work ranges.
void ReleaseMeshEditWorkRanges(state::Registry &, state::Entity mesh_entity);

// Rebuild dirty mesh layouts and mark bounds and display fields for recomputation.
void LayOutMeshes(state::Registry &, state::Entity viewport);
// The BoundsRun destroy handler, which queues the run's ranges for the settle pass to free and releases the mesh's pose namespaces.
void ReleaseMeshLayout(state::Registry &, state::Entity mesh_entity);
// Rederives each MeshDisplayDirty mesh's record display fields from the scene and its pose layout.
void RefreshMeshDisplays(state::Registry &, state::Entity viewport);
