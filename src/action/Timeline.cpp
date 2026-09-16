#include "action/Timeline.h"
#include "Variant.h"
#include "animation/AnimationTimeline.h"
#include "animation/Keyframes.h"
#include "gltf/SourceAssets.h"
#include "render/LightComponents.h"
#include "state/Scene.h"
#include "viewport/ViewportDisplay.h"

#include <cmath>
#include <optional>

namespace action::timeline {
namespace {
int WrapFrame(int frame, const TimelineRange &range) {
    if (frame >= range.StartFrame && frame <= range.EndFrame) return frame;
    const int size = range.EndFrame - range.StartFrame + 1;
    if (size <= 0) return frame;
    return range.StartFrame + ((frame - range.StartFrame) % size + size) % size;
}
} // namespace

void Apply(state::Scene &r, state::Entity viewport, const Action &action) {
    const auto enter_presentation = [&] {
        // Uses Rendered mode for scene lighting and material preview with the default world otherwise.
        const auto *source_assets = r.try_get<const gltf::SourceAssets>(viewport);
        const bool explicit_ibl = source_assets && source_assets->ImageBasedLight.has_value();
        const bool authored_lighting = explicit_ibl || !r.view<const LightIndex>().empty();
        const auto mode = authored_lighting ? ViewportShadingMode::Rendered : ViewportShadingMode::MaterialPreview;
        r.patch<ViewportDisplay>(viewport, [&](auto &s) { s.ViewportShading = s.FillMode = mode; s.ShowOverlays = false; });
        if (!authored_lighting && r.all_of<MaterialPreviewLighting>(viewport)) {
            r.patch<MaterialPreviewLighting>(viewport, [](auto &l) { l.Value.WorldOpacity = 1.f; });
        }
        if (authored_lighting && r.all_of<RenderedLighting>(viewport)) {
            r.patch<RenderedLighting>(viewport, [&](auto &l) {
                if (explicit_ibl) {
                    l.Value.BackgroundBlur = 0.f;
                } else {
                    l.Value.UseSceneWorld = false;
                    l.Value.WorldOpacity = 1.f;
                }
            });
        }
    };
    const auto set_frame = [&](int frame) {
        r.patch<TimelinePlayback>(viewport, [&](auto &p) { p.CurrentFrame = frame; });
        r.edit<PlaybackFrame>(viewport).Value = frame;
    };
    const auto step_frame = [&](int delta) {
        const int frame = r.get<const TimelinePlayback>(viewport).CurrentFrame + delta;
        set_frame(r.get<const TimelineNavigation>(viewport).Wrap ? WrapFrame(frame, r.get<const TimelineRange>(viewport)) : frame);
    };
    std::visit(
        overloaded{
            [&](EnterPresentation) { enter_presentation(); },
            [&](const TogglePlay &a) {
                r.patch<TimelinePlayback>(viewport, [&](auto &p) {
                    p.Playing = !p.Playing;
                    if (p.Playing) {
                        p.Reverse = a.Reverse;
                        p.PlayStartFrame = a.Frame;
                    }
                    p.CurrentFrame = a.Frame;
                });
                r.edit<PlaybackFrame>(viewport).Value = a.Frame;
            },
            [&](CancelPlay) {
                if (!r.get<const TimelinePlayback>(viewport).Playing) return;
                r.patch<TimelinePlayback>(viewport, [](auto &p) { p.Playing = false; p.CurrentFrame = p.PlayStartFrame; });
                r.edit<PlaybackFrame>(viewport).Value = r.get<const TimelinePlayback>(viewport).CurrentFrame;
            },
            [&](const SetFrame &a) { set_frame(a.Frame); },
            [&](const SetStartFrame &a) { r.patch<TimelineRange>(viewport, [&](auto &range) { range.StartFrame = a.Frame; }); },
            [&](const SetEndFrame &a) { r.patch<TimelineRange>(viewport, [&](auto &range) { range.EndFrame = a.Frame; }); },
            [&](JumpToStart) { JumpToStartFrame(r, viewport); },
            [&](JumpToEnd) { set_frame(r.get<const TimelineRange>(viewport).EndFrame); },
            [&](const OffsetFrame &a) { step_frame(a.Delta); },
            [&](const JumpTime &a) {
                const auto &nav = r.get<const TimelineNavigation>(viewport);
                const float frames = nav.JumpInSeconds ? nav.JumpDelta * r.get<const TimelineRange>(viewport).Fps : nav.JumpDelta;
                step_frame(int(std::lround(a.Backward ? -frames : frames)));
            },
            [&](const JumpKeyframe &a) {
                const auto keys = CollectKeyframes(r, viewport);
                const int current = r.get<const TimelinePlayback>(viewport).CurrentFrame;
                const auto &range = r.get<const TimelineRange>(viewport);
                const bool wrap = r.get<const TimelineNavigation>(viewport).Wrap;
                // Wrapping ignores keys outside the range and continues from the far end when none remain.
                const auto in_range = [&](float key) { return !wrap || (key >= float(range.StartFrame) && key <= float(range.EndFrame)); };
                const auto frame_of = [](float key) { return int(std::lround(key)); };
                std::optional<int> target;
                if (a.Next) {
                    for (const float key : keys) {
                        if (in_range(key) && frame_of(key) > current) {
                            target = frame_of(key);
                            break;
                        }
                    }
                    if (!target && wrap) {
                        if (const auto it = std::ranges::find_if(keys, in_range); it != keys.end()) target = frame_of(*it);
                    }
                } else {
                    for (auto it = keys.rbegin(); it != keys.rend(); ++it) {
                        if (in_range(*it) && frame_of(*it) < current) {
                            target = frame_of(*it);
                            break;
                        }
                    }
                    if (!target && wrap) {
                        if (const auto it = std::ranges::find_if(keys.rbegin(), keys.rend(), in_range); it != keys.rend()) target = frame_of(*it);
                    }
                }
                if (target) set_frame(*target);
            },
            [&](const SetNavigation &a) { r.replace<TimelineNavigation>(viewport, a.Value); },
            [&](const SetView &a) { r.replace<AnimationTimelineView>(viewport, AnimationTimelineView{a.PixelsPerFrame, a.ViewCenterFrame}); },
        },
        action
    );
}
} // namespace action::timeline
