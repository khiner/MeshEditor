#include "action/Timeline.h"
#include "Variant.h"
#include "gltf/SourceAssets.h"
#include "render/LightComponents.h"
#include "selection/SelectionState.h"
#include "state/Registry.h"
#include "viewport/ViewportEvents.h"

#include <cmath>

namespace action::timeline {
namespace {
int WrapFrame(int frame, const TimelineRange &range) {
    if (frame >= range.StartFrame && frame <= range.EndFrame) return frame;
    const int size = range.EndFrame - range.StartFrame + 1;
    if (size <= 0) return frame;
    return range.StartFrame + ((frame - range.StartFrame) % size + size) % size;
}
} // namespace

void Apply(state::Registry &r, const Action &action) {
    // The active viewport presents the scene in its next frame step, with its authored lighting when it has some.
    const auto enter_presentation = [&] {
        const auto *source_assets = r.try_get<const gltf::SourceAssets>(state::Document);
        const bool explicit_ibl = source_assets && source_assets->ImageBasedLight.has_value();
        const auto lighting = explicit_ibl                      ? PresentationLighting::SceneImageLight :
            !r.view<const LightIndex>().empty() ? PresentationLighting::SceneLights :
                                                  PresentationLighting::MaterialPreview;
        r.get_or_emplace<PendingViewSetup>(state::Document).Presentation = lighting;
    };
    const auto set_frame = [&](int frame) {
        r.patch<TimelinePlayback>(state::Document, [&](auto &p) { p.CurrentFrame = frame; });
        r.edit<PlaybackFrame>(state::Document).Value = frame;
    };
    const auto step_frame = [&](int delta) {
        const int frame = r.get<const TimelinePlayback>(state::Document).CurrentFrame + delta;
        set_frame(r.get<const TimelineNavigation>(state::Document).Wrap ? WrapFrame(frame, r.get<const TimelineRange>(state::Document)) : frame);
    };
    std::visit(
        overloaded{
            [&](EnterPresentation) { enter_presentation(); },
            [&](const TogglePlay &a) {
                r.patch<TimelinePlayback>(state::Document, [&](auto &p) {
                    p.Playing = !p.Playing;
                    if (p.Playing) {
                        p.Reverse = a.Reverse;
                        p.PlayStartFrame = a.Frame;
                    }
                    p.CurrentFrame = a.Frame;
                });
                r.edit<PlaybackFrame>(state::Document).Value = a.Frame;
            },
            [&](CancelPlay) {
                if (!r.get<const TimelinePlayback>(state::Document).Playing) return;
                r.patch<TimelinePlayback>(state::Document, [](auto &p) { p.Playing = false; p.CurrentFrame = p.PlayStartFrame; });
                r.edit<PlaybackFrame>(state::Document).Value = r.get<const TimelinePlayback>(state::Document).CurrentFrame;
            },
            [&](const SetFrame &a) { set_frame(a.Frame); },
            [&](const SetStartFrame &a) { r.patch<TimelineRange>(state::Document, [&](auto &range) { range.StartFrame = a.Frame; }); },
            [&](const SetEndFrame &a) { r.patch<TimelineRange>(state::Document, [&](auto &range) { range.EndFrame = a.Frame; }); },
            [&](JumpToStart) { JumpToStartFrame(r); },
            [&](JumpToEnd) { set_frame(r.get<const TimelineRange>(state::Document).EndFrame); },
            [&](const OffsetFrame &a) { step_frame(a.Delta); },
            [&](const JumpTime &a) {
                const auto &nav = r.get<const TimelineNavigation>(state::Document);
                const float frames = nav.JumpInSeconds ? nav.JumpDelta * r.get<const TimelineRange>(state::Document).Fps : nav.JumpDelta;
                step_frame(int(std::lround(a.Backward ? -frames : frames)));
            },
            [&](const JumpKeyframe &a) {
                const auto &keys = r.get<const SelectedKeyframes>(state::Document).Frames;
                const int current = r.get<const TimelinePlayback>(state::Document).CurrentFrame;
                const auto &range = r.get<const TimelineRange>(state::Document);
                const bool wrap = r.get<const TimelineNavigation>(state::Document).Wrap;
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
                    for (const auto key : std::views::reverse(keys)) {
                        if (in_range(key) && frame_of(key) < current) {
                            target = frame_of(key);
                            break;
                        }
                    }
                    if (!target && wrap) {
                        if (const auto it = std::ranges::find_if(keys.rbegin(), keys.rend(), in_range); it != keys.rend()) target = frame_of(*it);
                    }
                }
                if (target) set_frame(*target);
            },
            [&](const SetView &a) { r.replace<TimelineView>(state::Document, TimelineView{a.PixelsPerFrame, a.ViewCenterFrame}); },
        },
        action
    );
}
} // namespace action::timeline
