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
    Full,
    BlurredFull,
    BlurAccumulateFirst,
    BlurAccumulate,
    BlurResolve,
};

constexpr bool IsBlurAccumulate(RenderPhase p) { return p == RenderPhase::BlurAccumulateFirst || p == RenderPhase::BlurAccumulate; }

enum class SceneUpdate {
    Rebuild,
    Reuse,
};

void RecordRenderCommandBuffer(entt::registry &, entt::entity viewport, MTL::CommandBuffer *, SceneUpdate = SceneUpdate::Rebuild, RenderPhase = RenderPhase::Full);

// Records every blur step and the resolve into one command buffer using one view-UBO instance per step.
void RecordBlurStepsCommandBuffer(entt::registry &, entt::entity viewport, MTL::CommandBuffer *, std::span<const float> step_frames);

// Derive the listed mesh entities' base normals in one batched GPU submit-and-wait, writing the base normal stores.
// Meshes without triangles or adjacency are skipped.
// Call on the main thread between frames, where the per-frame derive buffers have no live GPU reader.
void DeriveBaseNormalsNow(entt::registry &, std::span<const entt::entity> mesh_entities);

// Complete the listed new or restored mesh entities' shading state.
// Derives base normals, encodes stored authored corner normals, and returns the authored-morph-shading gate.
// Call after the meshes' index buffers are written, under DeriveBaseNormalsNow's between-frames constraints.
void FinalizeNewMeshShadingNow(entt::registry &, std::span<const entt::entity> mesh_entities);

// Evaluates the final pending edit into canonical positions and affected normals in one submission.
// Returns the meshes whose positions changed.
std::vector<entt::entity> CommitPosedGeometry(entt::registry &, entt::entity viewport, std::span<const entt::entity> mesh_entities);
void ReleaseMeshEditWork(entt::registry &, entt::entity mesh_entity);

// Writes the posed-prelude dispatch counts for the next submission, or zeros when deform inputs are unchanged.
void SyncPreludeDispatchArgs(GpuBuffers &);
