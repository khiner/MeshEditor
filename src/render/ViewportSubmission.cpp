#include "render/ViewportSubmission.h"
#include "metal/AutoreleaseScope.h"

#include "Profile.h"
#include "mesh/MeshStore.h"
#include "render/GpuBuffers.h"
#include "state/Registry.h"
#include "viewport/Viewport.h"

void SubmitRecordedFrame(state::Registry &r, state::Entity viewport, mtl::Command &&command, RenderPhase phase) {
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
    auto &resources = r.Resources.get<ViewportRenderResources>();
    {
        const profile::CpuScope scope{"QueueSubmit"};
        resources.InFlight = ctx.Commit(std::move(command));
    }
    resources.RenderPending = true;
    r.Resources.get<MeshStore>().FrameSubmitted();
}

void RecordAndSubmitFrame(state::Registry &r, state::Entity viewport, RenderPhase phase) {
    const mtl::AutoreleaseScope native_scope;
    const auto &ctx = r.Resources.get<const mtl::Context>();
    auto command = ctx.Begin();
    RecordRenderCommandBuffer(r, viewport, command, phase);
    SubmitRecordedFrame(r, viewport, std::move(command), phase);
}

bool ViewportImageReady(const Viewport &viewport) {
    const auto extent = viewport.Targets.BuiltColorExtent();
    return extent.Width != 0 && extent.Height != 0;
}

void WaitForRender(state::Registry &r) {
    const mtl::AutoreleaseScope native_scope;
    auto &resources = r.Resources.get<ViewportRenderResources>();
    if (!resources.RenderPending) return;

    {
        const profile::CpuScope scope{"WaitGpu"};
        profile::Resolve(r.Resources.get<const mtl::Context>().Wait(resources.InFlight));
    }
    resources.InFlight = 0u;
    r.Resources.get<GpuBuffers>().Ctx.ReclaimRetiredBuffers();
    r.Resources.get<MeshStore>().FrameCompleted();
    r.Resources.get<const mtl::Context>().TrimPageCache();
    resources.RenderPending = false;
}
