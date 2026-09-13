#include "action/Timeline.h"
#include "Variant.h"
#include "animation/AnimationTimeline.h"
#include "gltf/SourceAssets.h"
#include "project/Registry.h"
#include "render/LightComponents.h"
#include "viewport/ViewportDisplay.h"

#include <entt/entity/registry.hpp>

namespace action::timeline {
void Apply(entt::registry &r, entt::entity viewport, const Action &action) {
    const auto enter_presentation = [&] {
        // Uses Rendered mode for scene lighting and material preview with the default world otherwise.
        const auto *source_assets = r.try_get<const gltf::SourceAssets>(viewport);
        const bool explicit_ibl = source_assets && source_assets->ImageBasedLight.has_value();
        const bool authored_lighting = explicit_ibl || !r.storage<LightIndex>().empty();
        const auto mode = authored_lighting ? ViewportShadingMode::Rendered : ViewportShadingMode::MaterialPreview;
        project::Patch<ViewportDisplay>(r, viewport, [&](auto &s) { s.ViewportShading = s.FillMode = mode; s.ShowOverlays = false; });
        if (!authored_lighting && r.all_of<MaterialPreviewLighting>(viewport)) {
            project::Patch<MaterialPreviewLighting>(r, viewport, [](auto &l) { l.WorldOpacity = 1.f; });
        }
        if (authored_lighting && r.all_of<RenderedLighting>(viewport)) {
            project::Patch<RenderedLighting>(r, viewport, [&](auto &l) {
                if (explicit_ibl) {
                    l.BackgroundBlur = 0.f;
                } else {
                    l.UseSceneWorld = false;
                    l.WorldOpacity = 1.f;
                }
            });
        }
    };
    std::visit(
        overloaded{
            [&](EnterPresentation) { enter_presentation(); },
            [&](StartPresentation) {
                enter_presentation();
                project::Patch<TimelinePlayback>(r, viewport, [](auto &p) { p.Playing = true; });
            },
            [&](const TogglePlay &a) {
                project::Patch<TimelinePlayback>(r, viewport, [&](auto &p) { p.Playing = !p.Playing; p.CurrentFrame = a.Frame; });
                r.get<PlaybackFrame>(viewport).Value = a.Frame;
            },
            [&](const SetFrame &a) {
                project::Patch<TimelinePlayback>(r, viewport, [&](auto &p) { p.CurrentFrame = a.Frame; });
                r.get<PlaybackFrame>(viewport).Value = a.Frame;
            },
            [&](const SetStartFrame &a) { project::Patch<TimelineRange>(r, viewport, [&](auto &range) { range.StartFrame = a.Frame; }); },
            [&](const SetEndFrame &a) { project::Patch<TimelineRange>(r, viewport, [&](auto &range) { range.EndFrame = a.Frame; }); },
            [&](JumpToStart) { JumpToStartFrame(r, viewport); },
            [&](JumpToEnd) {
                const auto frame = r.get<const TimelineRange>(viewport).EndFrame;
                project::Patch<TimelinePlayback>(r, viewport, [&](auto &p) { p.CurrentFrame = frame; });
                r.get<PlaybackFrame>(viewport).Value = frame;
            },
            [&](const SetView &a) { project::Replace<AnimationTimelineView>(r, viewport, AnimationTimelineView{a.PixelsPerFrame, a.ViewCenterFrame}); },
        },
        action
    );
}
} // namespace action::timeline
