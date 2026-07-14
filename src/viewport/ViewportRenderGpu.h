#pragma once

#include <entt/entity/fwd.hpp>
#include <vulkan/vulkan.hpp>

struct DrawListBuilder;
struct DrawBufferPair;

// Upload the draw list to the per-pass buffers, grow the identity index buffer if needed,
// and flush any deferred descriptor updates accumulated during buffer growth.
void FlushDrawList(entt::registry &, vk::Device, const DrawListBuilder &, DrawBufferPair &);

// Which parts of a frame one recording covers.
enum class RenderPhase {
    Full, // Scene and overlays together: the normal render.
    SceneAccumulate, // Motion blur only: shades the scene at one sub-frame time and sums it into the blur target.
    ResolveOverlays, // Motion blur only: averages the summed sub-frames, then draws scene depth and overlays over them.
};

// Build the main draw list (or just the silhouette portion) into the DrawState component and record the render pass.
void RecordRenderCommandBuffer(entt::registry &, entt::entity viewport, vk::CommandBuffer, bool silhouette_only = false, RenderPhase = RenderPhase::Full);

// Zero the motion blur accumulation target. One-shot recording into the provided command buffer.
void RecordMotionBlurClear(entt::registry &, vk::CommandBuffer);

#ifdef MVK_FORCE_STAGED_TRANSFERS
void RecordTransferCommandBuffer(entt::registry &, entt::entity viewport, vk::CommandBuffer);
#endif
