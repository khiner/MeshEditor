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
    BlurredFull, // Motion blur, single step: shades the scene, blurs it across the whole shutter, and draws overlays sharp over it.
    BlurAccumulateFirst, // Motion blur, multi-step: the first step, which clears the blur target it sums into.
    BlurAccumulate, // Motion blur, multi-step: shades the scene at one step's centre, blurs it along that step's screen motion, and sums it into the blur target.
    BlurResolve, // Motion blur, multi-step: averages the summed steps, then draws scene depth and overlays over them.
};

constexpr bool IsBlurAccumulate(RenderPhase p) { return p == RenderPhase::BlurAccumulateFirst || p == RenderPhase::BlurAccumulate; }

// Build the main draw list (or just the silhouette portion) into the DrawState component and record the render pass.
void RecordRenderCommandBuffer(entt::registry &, entt::entity viewport, vk::CommandBuffer, bool silhouette_only = false, RenderPhase = RenderPhase::Full);

#ifdef MVK_FORCE_STAGED_TRANSFERS
void RecordTransferCommandBuffer(entt::registry &, entt::entity viewport, vk::CommandBuffer);
#endif
