#pragma once

#include "action/Timeline.h"

#include <memory>
#include <optional>

struct SvgResource;

struct AnimationTimeline {
    int CurrentFrame{1};
    int StartFrame{1}, EndFrame{100};
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

std::optional<action::timeline::Action> RenderAnimationTimeline(const AnimationTimeline &, AnimationTimelineView &, const AnimationIcons &);
