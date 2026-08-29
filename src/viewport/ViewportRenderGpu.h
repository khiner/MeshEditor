#pragma once

#include "Range.h"
#include "metal/MetalCpp.h"
#include "metal/Slots.h"

#include <entt/entity/fwd.hpp>
#include <span>
#include <vector>

namespace mtl {
struct BindlessSet;
struct Buffer;
struct PassChain;
} // namespace mtl

struct GpuBuffers;
struct MeshStore;
struct Pipelines;

enum class MeshletRouteMode : uint32_t { Single,
                                         Material,
                                         Transmission,
                                         Visibility };

struct MeshletCullConfig {
    MeshletRouteMode Mode{MeshletRouteMode::Single};
    uint32_t RequiredInstanceFlags{0};
    uint32_t RouteMask{0x3ffu};
    uint32_t UboOffset{0};
    uint32_t PyramidSamplerSlot{InvalidSlot};
    bool SortBlend{false};
    bool TwoPhase{false};
};

void RecordMeshletCull(mtl::PassChain &, const mtl::BindlessSet &, const Pipelines &, GpuBuffers &, MeshletCullConfig);
void RecordOverlayJobCull(
    mtl::PassChain &, const mtl::BindlessSet &, const Pipelines &, GpuBuffers &,
    bool extras_only = false, uint32_t ubo_offset = 0
);
void DrawOverlayJobs(MTL::RenderCommandEncoder *, const GpuBuffers &, const MeshStore &);
// Rasterize the current meshlet routes into the visibility/depth pair shared by shading and selection.
void RecordMeshletVisibilityPass(
    mtl::PassChain &, const mtl::BindlessSet &, const Pipelines &, GpuBuffers &,
    bool transmission = false, uint32_t ubo_offset = 0
);
void RecordSilhouetteDepthPass(mtl::PassChain &, const mtl::BindlessSet &, const Pipelines &, GpuBuffers &, bool draw_meshlets, uint32_t ubo_offset = 0);
void DrawMeshlets(
    MTL::RenderCommandEncoder *, const GpuBuffers &, uint32_t route,
    uint32_t required_instance_flags = 0, uint32_t mesh_threads = 160u,
    uint32_t edit_edge_corner = 0u, uint32_t instance_filter = InvalidOffset
);

// Which parts of a frame one recording covers.
enum class RenderPhase {
    Full, // Scene and overlays together: the normal render.
    BlurredFull, // Motion blur, single step: shades the scene, blurs it across the whole shutter, and draws overlays sharp over it.
    BlurAccumulateFirst, // Motion blur, multi-step: the first step, which clears the blur target it sums into.
    BlurAccumulate, // Motion blur, multi-step: shades the scene at one step's centre, blurs it along that step's screen motion, and sums it into the blur target.
    BlurResolve, // Motion blur, multi-step: averages the summed steps, then draws scene depth and overlays over them.
};

constexpr bool IsBlurAccumulate(RenderPhase p) { return p == RenderPhase::BlurAccumulateFirst || p == RenderPhase::BlurAccumulate; }

// Whether this recording refreshes persistent GPU scene descriptors or reuses them unchanged.
enum class SceneUpdate {
    Rebuild,
    Reuse,
};

void RecordRenderCommandBuffer(entt::registry &, entt::entity viewport, MTL::CommandBuffer *, SceneUpdate = SceneUpdate::Rebuild, RenderPhase = RenderPhase::Full);

// Record every motion blur step and the resolve into one command buffer, each step reading its own
// view UBO instance (i + 1) by dynamic offset. `step_frames` holds each step's centre playback frame.
void RecordBlurStepsCommandBuffer(entt::registry &, entt::entity viewport, MTL::CommandBuffer *, std::span<const float> step_frames);

// Derive the listed mesh entities' base normals in one batched GPU submit-and-wait, writing the base normal stores.
// Meshes without triangles or adjacency are skipped.
// Call on the main thread between frames, where the per-frame derive buffers have no live GPU reader.
void DeriveBaseNormalsNow(entt::registry &, std::span<const entt::entity> mesh_entities);

// Complete the listed new or restored mesh entities' shading state.
// Derives base normals, encodes stashed authored corner normals, and decides the authored-morph-shading gate.
// Call after the meshes' index buffers are written, under DeriveBaseNormalsNow's between-frames constraints.
void FinalizeNewMeshShadingNow(entt::registry &, std::span<const entt::entity> mesh_entities);

// Commit posed positions and normals with one submit. Returns the meshes whose positions changed.
std::vector<entt::entity> CommitPosedGeometry(entt::registry &, std::span<const entt::entity> mesh_entities);

// Write the posed prelude's indirect dispatch group counts for the next submit.
// Deform-input changes since the last submit select the recorded counts, and an unchanged pose selects zeros.
void SyncPreludeDispatchArgs(GpuBuffers &);
