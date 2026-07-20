#pragma once

#include "vulkan/Slots.h"

#include <entt/entity/fwd.hpp>
#include <vulkan/vulkan.hpp>

struct DrawListBuilder;
struct DrawBufferPair;
struct GpuBuffers;
struct Pipelines;

// Upload the draw list to the per-pass buffers, grow the identity index buffer if needed,
// and flush any deferred descriptor updates accumulated during buffer growth.
void FlushDrawList(entt::registry &, vk::Device, const DrawListBuilder &, DrawBufferPair &);

// Zero `pair`'s indirect instance counts, then refill region A and the visible-index remap from
// per-instance bounds tested against the view frustum. With `visibility_slot`, region A holds only
// previously-visible instances, and the caller must run the occlusion pass and draw region B.
void RecordFrustumCull(vk::CommandBuffer, const Pipelines &, const GpuBuffers &, const DrawBufferPair &, const DrawListBuilder &, uint32_t ubo_offset = 0, uint32_t visibility_slot = InvalidSlot);

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
    Rebuild, // Build the whole draw list anew.
    SilhouetteOnly, // Keep the main portion, rebuild only the silhouette batch.
    Reuse, // Record from the list as it stands.
};

// Build the main draw list (or just the silhouette portion) into the DrawState component and record the render pass.
void RecordRenderCommandBuffer(entt::registry &, entt::entity viewport, vk::CommandBuffer, DrawListUse = DrawListUse::Rebuild, RenderPhase = RenderPhase::Full);

// Record every motion blur step and the resolve into one command buffer, each step reading its own
// view UBO instance (i + 1) by dynamic offset. `step_frames` holds each step's centre playback frame.
void RecordBlurStepsCommandBuffer(entt::registry &, entt::entity viewport, vk::CommandBuffer, std::span<const float> step_frames);

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

#ifdef MVK_FORCE_STAGED_TRANSFERS
void RecordTransferCommandBuffer(entt::registry &, entt::entity viewport, vk::CommandBuffer);
#endif
