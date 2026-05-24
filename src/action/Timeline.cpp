#include "action/Timeline.h"
#include "AnimationTimeline.h"
#include "Variant.h"
#include "ViewportComponents.h"

namespace action::timeline {
void Apply(entt::registry &r, entt::entity viewport, const Action &action) {
    std::visit(
        overloaded{
            [&](Play) {
                r.patch<ViewportDisplay>(viewport, [](auto &s) { s.ViewportShading = s.FillMode = ViewportShadingMode::MaterialPreview; s.ShowOverlays = false; });
                r.patch<TimelinePlayback>(viewport, [](auto &p) { p.Playing = !p.Playing; });
            },
            [&](TogglePlay) { r.patch<TimelinePlayback>(viewport, [](auto &p) { p.Playing = !p.Playing; }); },
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
