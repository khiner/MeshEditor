#pragma once

#include <entt/entity/fwd.hpp>
#include <vulkan/vulkan.hpp>

struct DrawListBuilder;
struct DrawBufferPair;

// Upload the draw list to the per-pass buffers, grow the identity index buffer if needed,
// and flush any deferred descriptor updates accumulated during buffer growth.
void FlushDrawList(entt::registry &, entt::entity scene_entity, vk::Device, const DrawListBuilder &, DrawBufferPair &);

// Build the main draw list (or just the silhouette portion) into the SceneDrawState component and record the render pass.
void RecordRenderCommandBuffer(entt::registry &, entt::entity scene_entity, vk::CommandBuffer, bool silhouette_only = false);

#ifdef MVK_FORCE_STAGED_TRANSFERS
void RecordTransferCommandBuffer(entt::registry &, entt::entity scene_entity, vk::CommandBuffer);
#endif
