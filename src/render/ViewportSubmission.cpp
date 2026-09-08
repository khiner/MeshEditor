#include "render/ViewportSubmission.h"

#include "Profile.h"
#include "render/GpuBuffers.h"
#include "render/Pipelines.h"
#include "render/Textures.h"
#include "viewport/FrameState.h"
#include "viewport/RenderExtent.h"
#include "viewport/Viewport.h"
#include "viewport/ViewportDisplay.h"
#include <Metal/MTLCommandQueue.hpp>
#include <entt/entity/registry.hpp>
// Dispatch sizes follow scene recording because the rebuild determines their counts.
void SubmitRecordedFrame(entt::registry &r, MTL::CommandBuffer *command_buffer) {
    const auto &ctx = r.ctx().get<const mtl::Context>();
    auto &buffers = r.ctx().get<GpuBuffers>();
    SyncPreludeDispatchArgs(buffers);
    ctx.CommitResidency();
    {
        const profile::CpuScope scope{"QueueSubmit"};
        command_buffer->commit();
    }
    r.ctx().get<ViewportRenderResources>().InFlight = command_buffer;
    r.ctx().get<FrameState>().RenderPending = true;
}

void RecordAndSubmitFrame(entt::registry &r, entt::entity viewport, SceneUpdate update, RenderPhase phase) {
    const auto &ctx = r.ctx().get<const mtl::Context>();
    auto &resources = r.ctx().get<ViewportRenderResources>();
    auto *command_buffer = ctx.Queue->commandBuffer();
    RecordRenderCommandBuffer(r, viewport, command_buffer, update, phase);
    resources.RecordedPhase = phase;
    SubmitRecordedFrame(r, command_buffer);
}

bool ViewportImageReady(const entt::registry &r) {
    const auto extent = r.ctx().get<const Pipelines>().BuiltColorExtent();
    return extent.Width != 0 && extent.Height != 0;
}

void SetStudioEnvironment(entt::registry &r, uint32_t index) {
    const auto &ctx = r.ctx().get<const mtl::Context>();
    const auto &pipelines = r.ctx().get<const Pipelines>();
    auto &slots = r.ctx().get<mtl::BindlessSet>();
    auto &environments = r.ctx().get<EnvironmentStore>();
    auto &hdri = environments.Hdris[index];
    if (!hdri.Prefiltered) {
        hdri.Prefiltered = CreateIblFromHdri(ctx, slots, pipelines.IblPrefilter, hdri.Path, hdri.Name);
    }
    const auto &pre = *hdri.Prefiltered;
    environments.ActiveHdriIndex = index;
    environments.StudioWorld = {.Ibl = MakeIblSamplers(pre, environments), .Name = hdri.Name};
}

void RebuildStudioEnvironments(entt::registry &r) {
    auto &slots = r.ctx().get<mtl::BindlessSet>();
    auto &environments = r.ctx().get<EnvironmentStore>();
    if (environments.Hdris.empty()) return; // No studio environment to index into.
    const auto release = [&slots](uint32_t sampler_slot) {
        if (sampler_slot != InvalidSlot) slots.Release({SlotType::CubeSampler, sampler_slot});
    };
    for (auto &hdri : environments.Hdris) {
        if (!hdri.Prefiltered) continue;
        release(hdri.Prefiltered->DiffuseEnv.SamplerSlot);
        release(hdri.Prefiltered->SpecularEnv.SamplerSlot);
        hdri.Prefiltered.reset();
    }
    SetStudioEnvironment(r, environments.ActiveHdriIndex);
}

void SetStudioEnvironment(entt::registry &r, std::string_view name) {
    const auto &hdris = r.ctx().get<const EnvironmentStore>().Hdris;
    const auto it = std::ranges::find(hdris, name, &HdriEntry::Name);
    SetStudioEnvironment(r, it != hdris.end() ? uint32_t(std::distance(hdris.begin(), it)) : 0u);
}

void WaitForRender(entt::registry &r) {
    auto &frame = r.ctx().get<FrameState>();
    if (!frame.RenderPending) return;

    auto &resources = r.ctx().get<ViewportRenderResources>();
    if (resources.InFlight) {
        const profile::CpuScope scope{"WaitGpu"};
        resources.InFlight->waitUntilCompleted();
    }
    profile::Resolve(resources.InFlight);
    resources.InFlight = nullptr;
    r.ctx().get<GpuBuffers>().Ctx.ReclaimRetiredBuffers();
    frame.RenderPending = false;
}
