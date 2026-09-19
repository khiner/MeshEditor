#include "render/ViewportSubmission.h"

#include "Profile.h"
#include "render/GpuBuffers.h"
#include "render/RenderTargets.h"
#include "state/Scene.h"
#include "viewport/FrameState.h"
#include "viewport/RenderExtent.h"
#include "viewport/Viewport.h"
#include <Metal/MTLCommandQueue.hpp>
// Dispatch sizes follow scene recording because the rebuild determines their counts.
void SubmitRecordedFrame(state::Scene &r, MTL::CommandBuffer *command_buffer) {
    const auto &ctx = r.Context.get<const mtl::Context>();
    auto &buffers = r.Context.get<GpuBuffers>();
    SyncPreludeDispatchArgs(buffers);
    ctx.CommitResidency();
    {
        const profile::CpuScope scope{"QueueSubmit"};
        command_buffer->commit();
    }
    r.Context.get<ViewportRenderResources>().InFlight = command_buffer;
    r.Context.get<FrameState>().RenderPending = true;
}

void RecordAndSubmitFrame(state::Scene &r, state::Entity viewport, SceneUpdate update, RenderPhase phase) {
    const auto &ctx = r.Context.get<const mtl::Context>();
    auto &resources = r.Context.get<ViewportRenderResources>();
    auto *command_buffer = ctx.Queue->commandBuffer();
    RecordRenderCommandBuffer(r, viewport, command_buffer, update, phase);
    resources.RecordedPhase = phase;
    SubmitRecordedFrame(r, command_buffer);
}

bool ViewportImageReady(const state::Scene &r) {
    const auto extent = r.Context.get<const RenderTargets>().BuiltColorExtent();
    return extent.Width != 0 && extent.Height != 0;
}

void WaitForRender(state::Scene &r) {
    auto &frame = r.Context.get<FrameState>();
    if (!frame.RenderPending) return;

    auto &resources = r.Context.get<ViewportRenderResources>();
    if (resources.InFlight) {
        const profile::CpuScope scope{"WaitGpu"};
        resources.InFlight->waitUntilCompleted();
    }
    profile::Resolve(resources.InFlight);
    resources.InFlight = nullptr;
    r.Context.get<GpuBuffers>().Ctx.ReclaimRetiredBuffers();
    frame.RenderPending = false;
}
