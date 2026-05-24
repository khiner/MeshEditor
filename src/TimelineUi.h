#pragma once

#include "AnimationTimeline.h"
#include "action/Timeline.h"

#include <optional>

struct AnimationIcons;

std::optional<action::timeline::Action> RenderAnimationTimeline(const TimelineRange &, const TimelinePlayback &, const AnimationTimelineView &, const AnimationIcons &);
