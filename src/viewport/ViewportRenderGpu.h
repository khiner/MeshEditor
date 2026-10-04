#pragma once

#include "numeric/uvec2.h"

#include "Range.h"
#include "gpu/MeshletRouteMode.h"
#include "metal/Slots.h"

#include "state/Entity.h"
#include <optional>
#include <span>
#include <vector>

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
// Records the recompute of normals and edit work for restored vertex ranges without mutating the vertices.
// The chain submits where stale footprints size the edit work, and the refresh runs with its next submit.
// One refresh records per chain submit.
void RefreshEditedPositions(state::Scene &, mtl::ComputeChain &, std::span<const MeshVertexChanges>);

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
    MTL::RenderCommandEncoder *, const state::Scene &, state::Entity mesh_entity, uint32_t instance, bool sound_points = false,
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

enum class SceneUpdate {
    Rebuild,
    Reuse,
};

void RecordRenderCommandBuffer(state::Scene &, state::Entity viewport, MTL::CommandBuffer *, SceneUpdate = SceneUpdate::Rebuild, RenderPhase = RenderPhase::Full);

// Records every blur step and the resolve into one command buffer using one view-UBO instance per step.
void RecordBlurStepsCommandBuffer(state::Scene &, state::Entity viewport, MTL::CommandBuffer *, std::span<const uint32_t> sample_weights);

// Resolves whether the listed mesh entities retain authored shading normals under morphing.
// The CPU resolves targets with authored normal deltas.
// Position-only targets derive their full-weight poses on the chain, which submits once for them.
// The derived pose tests whether derivation moves the normals authored shading would pin.
// Call after the base derive has run, since the pin test compares against the base normal stores.
void UpdateAuthoredMorphShading(state::Scene &, mtl::ComputeChain &, std::span<const state::Entity> mesh_entities);

// Completes the listed new or restored mesh entities' shading state on the chain.
// Derives base normals, encodes stored authored corner normals, and resolves the authored-morph-shading gate.
// The chain submits where the host reads derived normals, and otherwise the derive runs with its next submit.
// Call after canonical connectivity is available.
void FinalizeNewMeshShading(state::Scene &, mtl::ComputeChain &, std::span<const state::Entity> mesh_entities);

// Evaluates the final pending edit into canonical positions and affected normals, submitting the chain to read which changed.
// The refits and selection refresh then record on the chain, and the selection summaries publish once it submits again.
// Returns the meshes whose positions changed.
std::vector<state::Entity> CommitPosedGeometry(state::Scene &, mtl::ComputeChain &, state::Entity viewport, std::span<const state::Entity> mesh_entities);
void ReleaseMeshEditWork(state::Scene &, state::Entity mesh_entity);

// Writes the posed-prelude dispatch counts for the next submission, or zeros when deform inputs are unchanged.
void SyncPreludeDispatchArgs(GpuBuffers &);

// Rederives the listed meshes' record display fields from the scene, and returns whether one needs the scene layout rebuilt instead, as a posed mesh does.
bool RefreshMeshDisplays(state::Scene &, state::Entity viewport, std::span<const state::Entity> mesh_entities);
