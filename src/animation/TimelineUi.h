#pragma once

#include "action/Timeline.h"
#include "animation/AnimationTimeline.h"

#include <optional>

struct AnimationIcons;

std::optional<action::timeline::Action> RenderAnimationTimeline(const TimelineRange &, const TimelinePlayback &, const AnimationTimelineView &, const AnimationIcons &);
