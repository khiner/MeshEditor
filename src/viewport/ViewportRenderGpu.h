#pragma once

#include "gpu/ExtrasLinePushConstants.h"
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

struct DrawListBuilder;
struct GpuBuffers;
struct InstanceArena;
struct Pipelines;

enum class MeshletRouteMode : uint32_t { Single,
                                         Material,
                                         Transmission,
                                         Visibility };

struct MeshletCullConfig {
    MeshletRouteMode Mode{MeshletRouteMode::Single};
    uint32_t RequiredInstanceFlags{0};
    uint32_t UboOffset{0};
    uint32_t PyramidSamplerSlot{InvalidSlot};
    bool SortBlend{false};
    bool TwoPhase{false};
};

// Upload the draw list to the pass's draw-data buffer, flushing any deferred bindless updates
// accumulated during buffer growth.
void FlushDrawList(entt::registry &, const DrawListBuilder &, mtl::Buffer &draw_data);

// Every extras line source this frame (gizmos and collision shape wireframes), with the dispatch each one needs.
std::vector<ExtrasLinePushConstants> CollectExtrasLines(const entt::registry &, const InstanceArena &);
void RecordMeshletCull(mtl::PassChain &, const mtl::BindlessSet &, const Pipelines &, GpuBuffers &, MeshletCullConfig);
void RecordSilhouetteDepthPass(mtl::PassChain &, const mtl::BindlessSet &, const Pipelines &, GpuBuffers &, bool draw_meshlets, uint32_t ubo_offset = 0);
void DrawMeshlets(MTL::RenderCommandEncoder *, const GpuBuffers &, uint32_t route, uint32_t required_instance_flags = 0);

// Which parts of a frame one recording covers.
enum class RenderPhase {
    Full, // Scene and overlays together: the normal render.
    BlurredFull, // Motion blur, single step: shades the scene, blurs it across the whole shutter, and draws overlays sharp over it.
    BlurAccumulateFirst, // Motion blur, multi-step: the first step, which clears the blur target it sums into.
    BlurAccumulate, // Motion blur, multi-step: shades the scene at one step's centre, blurs it along that step's screen motion, and sums it into the blur target.
    BlurResolve, // Motion blur, multi-step: averages the summed steps, then draws scene depth and overlays over them.
};

constexpr bool IsBlurAccumulate(RenderPhase p) { return p == RenderPhase::BlurAccumulateFirst || p == RenderPhase::BlurAccumulate; }

// How a recording treats the DrawState draw list.
enum class DrawListUse {
    Rebuild,
    Reuse,
};

void RecordRenderCommandBuffer(entt::registry &, entt::entity viewport, MTL::CommandBuffer *, DrawListUse = DrawListUse::Rebuild, RenderPhase = RenderPhase::Full);

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

// Copy the mesh's posed positions and derived normals from the last submitted frame (fenced complete) into the canonical stores.
// Returns true when any position changed.
bool CommitPosedGeometry(entt::registry &, entt::entity mesh_entity);

// Write the posed prelude's indirect dispatch group counts for the next submit.
// Deform-input changes since the last submit select the recorded counts, and an unchanged pose selects zeros.
void SyncPreludeDispatchArgs(GpuBuffers &);
