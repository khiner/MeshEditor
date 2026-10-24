#include "render/ViewportSubmission.h"
#include "metal/AutoreleaseScope.h"

#include "Profile.h"
#include "mesh/MeshStore.h"
#include "render/GpuBuffers.h"
#include "state/Registry.h"
#include "viewport/Viewport.h"
#include <Metal/MTLCommandQueue.hpp>
ViewportRenderResources::ViewportRenderResources() = default;
ViewportRenderResources::~ViewportRenderResources() { mtl::AutoreleaseScope::Release(InFlight); }

void SubmitRecordedFrame(state::Registry &r, state::Entity viewport, MTL::CommandBuffer *command_buffer, RenderPhase phase) {
    const mtl::AutoreleaseScope native_scope;
    const auto &ctx = r.Resources.get<const mtl::Context>();
    const auto scene = SceneOf(r, viewport);
    auto &live = r.edit<Viewport>(viewport);
    live.OcclusionStale = false;
    if (phase != RenderPhase::Prepare) {
        live.RecordedPhase = phase;
        live.RenderedScene = scene;
        live.RenderRequested = false;
    }
    ctx.CommitResidency();
    {
        const profile::CpuScope scope{"QueueSubmit"};
        mtl::Commit(command_buffer);
    }
    auto &resources = r.Resources.get<ViewportRenderResources>();
    resources.InFlight = NS::RetainPtr(command_buffer);
    resources.RenderPending = true;
    r.Resources.get<MeshStore>().FrameSubmitted();
}

void RecordAndSubmitFrame(state::Registry &r, state::Entity viewport, RenderPhase phase) {
    const mtl::AutoreleaseScope native_scope;
    const auto &ctx = r.Resources.get<const mtl::Context>();
    auto *command_buffer = ctx.Queue->commandBuffer();
    RecordRenderCommandBuffer(r, viewport, command_buffer, phase);
    SubmitRecordedFrame(r, viewport, command_buffer, phase);
}

bool ViewportImageReady(const Viewport &viewport) {
    const auto extent = viewport.Targets.BuiltColorExtent();
    return extent.Width != 0 && extent.Height != 0;
}

void WaitForRender(state::Registry &r) {
    const mtl::AutoreleaseScope native_scope;
    auto &resources = r.Resources.get<ViewportRenderResources>();
    if (!resources.RenderPending) return;

    if (resources.InFlight) {
        const profile::CpuScope scope{"WaitGpu"};
        resources.InFlight->waitUntilCompleted();
    }
    profile::Resolve(resources.InFlight.get());
    resources.InFlight = nullptr;
    r.Resources.get<GpuBuffers>().Ctx.ReclaimRetiredBuffers();
    r.Resources.get<MeshStore>().FrameCompleted();
    r.Resources.get<const mtl::Context>().TrimPageCache();
    resources.RenderPending = false;
}
