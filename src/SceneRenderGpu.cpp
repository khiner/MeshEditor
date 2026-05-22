#include "SceneRenderGpu.h"

#include "SceneDrawState.h"
#include "scene_impl/SceneBuffers.h"

#include <entt/entity/registry.hpp>

#ifdef MVK_FORCE_STAGED_TRANSFERS
#include "Timer.h"
#endif

void FlushDrawList(entt::registry &R, entt::entity scene_entity, vk::Device device, const DrawListBuilder &draw_list, DrawBufferPair &pair) {
    auto &buffers = R.get<SceneBuffers>(scene_entity);
    if (!draw_list.Draws.empty()) pair.DrawData.Update(as_bytes(draw_list.Draws));
    if (!draw_list.IndirectCommands.empty()) pair.Indirect.Update(as_bytes(draw_list.IndirectCommands));
    buffers.EnsureIdentityIndexBuffer(draw_list.MaxIndexCount);
    if (auto descriptor_updates = buffers.Ctx.GetDeferredDescriptorUpdates(); !descriptor_updates.empty()) {
        device.updateDescriptorSets(std::move(descriptor_updates), {});
        buffers.Ctx.ClearDeferredDescriptorUpdates();
    }
}

#ifdef MVK_FORCE_STAGED_TRANSFERS
void RecordTransferCommandBuffer(entt::registry &R, entt::entity scene_entity, vk::CommandBuffer cb) {
    auto &buffers = R.get<SceneBuffers>(scene_entity);
    const Timer timer{"RecordTransferCommandBuffer"};
    cb.reset({});
    cb.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    buffers.Ctx.RecordDeferredCopies(cb);
    cb.end();
}
#endif
