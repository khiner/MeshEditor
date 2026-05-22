#pragma once

#include <entt/entity/fwd.hpp>
#include <vulkan/vulkan.hpp>

struct DrawListBuilder;
struct DrawBufferPair;

// Upload the draw list to the per-pass buffers, grow the identity index buffer if needed,
// and flush any deferred descriptor updates accumulated during buffer growth.
void FlushDrawList(entt::registry &, entt::entity scene_entity, vk::Device, const DrawListBuilder &, DrawBufferPair &);

#ifdef MVK_FORCE_STAGED_TRANSFERS
void RecordTransferCommandBuffer(entt::registry &, entt::entity scene_entity, vk::CommandBuffer);
#endif
