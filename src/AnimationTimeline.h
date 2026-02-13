#pragma once

#include <memory>
#include <optional>
#include <variant>

struct SvgResource;

struct AnimationTimeline {
    int CurrentFrame{1};
    int StartFrame{1};
    int EndFrame{100};
    float Fps{24.0f};
    bool Playing{false};
};

struct AnimationTimelineView {
    float PixelsPerFrame{8.0f};
    float ViewCenterFrame{50.0f}; // Frame at horizontal center of visible scroll region
};

struct AnimationIcons {
    std::unique_ptr<SvgResource> Play, Pause, JumpStart, JumpEnd;
};

namespace timeline_action {
struct TogglePlay {};
struct SetFrame {
    int Frame;
};
struct SetStartFrame {
    int Frame;
};
struct SetEndFrame {
    int Frame;
};
struct JumpToStart {};
struct JumpToEnd {};
} // namespace timeline_action

using AnimationTimelineAction = std::variant<
    timeline_action::TogglePlay, timeline_action::SetFrame,
    timeline_action::SetStartFrame, timeline_action::SetEndFrame,
    timeline_action::JumpToStart, timeline_action::JumpToEnd>;

std::optional<AnimationTimelineAction> RenderAnimationTimeline(const AnimationTimeline &, AnimationTimelineView &, const AnimationIcons &);
