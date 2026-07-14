#pragma once

#include "action/Timeline.h"
#include "animation/AnimationTimeline.h"

#include <optional>

struct AnimationIcons;

// Sets `scrubbing` true while the frame marker is held (mouse down in the header).
std::optional<action::timeline::Action> RenderAnimationTimeline(const TimelineRange &, const TimelinePlayback &, const AnimationTimelineView &, const AnimationIcons &, bool &scrubbing);
