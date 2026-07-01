#include "action/Timeline.h"
#include "Variant.h"
#include "animation/AnimationTimeline.h"
#include "gltf/SourceAssets.h"
#include "viewport/ViewportDisplay.h"

#include <entt/entity/registry.hpp>

namespace action::timeline {
void Apply(entt::registry &r, entt::entity viewport, const Action &action) {
    const auto enter_presentation = [&] {
        r.patch<ViewportDisplay>(viewport, [](auto &s) { s.ViewportShading = s.FillMode = ViewportShadingMode::MaterialPreview; s.ShowOverlays = false; });
        // If the scene doesn't have an IBL, show the default IBL during presentation.
        const auto *source_assets = r.try_get<const gltf::SourceAssets>(viewport);
        const bool explicit_ibl = source_assets && source_assets->ImageBasedLight.has_value();
        if (!explicit_ibl && r.all_of<MaterialPreviewLighting>(viewport)) {
            r.patch<MaterialPreviewLighting>(viewport, [](auto &l) { l.WorldOpacity = 1.f; });
        }
    };
    std::visit(
        overloaded{
            [&](EnterPresentation) { enter_presentation(); },
            [&](StartPresentation) {
                enter_presentation();
                r.patch<TimelinePlayback>(viewport, [](auto &p) { p.Playing = true; });
            },
            [&](const TogglePlay &a) {
                r.patch<TimelinePlayback>(viewport, [&](auto &p) { p.Playing = !p.Playing; p.CurrentFrame = a.Frame; });
                r.get<PlaybackFrame>(viewport).Value = a.Frame;
            },
            [&](const SetFrame &a) {
                r.patch<TimelinePlayback>(viewport, [&](auto &p) { p.CurrentFrame = a.Frame; });
                r.get<PlaybackFrame>(viewport).Value = a.Frame;
            },
            [&](const SetStartFrame &a) { r.patch<TimelineRange>(viewport, [&](auto &range) { range.StartFrame = a.Frame; }); },
            [&](const SetEndFrame &a) { r.patch<TimelineRange>(viewport, [&](auto &range) { range.EndFrame = a.Frame; }); },
            [&](JumpToStart) { JumpToStartFrame(r, viewport); },
            [&](JumpToEnd) {
                const auto frame = r.get<const TimelineRange>(viewport).EndFrame;
                r.patch<TimelinePlayback>(viewport, [&](auto &p) { p.CurrentFrame = frame; });
                r.get<PlaybackFrame>(viewport).Value = frame;
            },
            [&](const SetView &a) { r.replace<AnimationTimelineView>(viewport, AnimationTimelineView{a.PixelsPerFrame, a.ViewCenterFrame}); },
        },
        action
    );
}
} // namespace action::timeline
