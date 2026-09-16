#pragma once

#include "action/Timeline.h"
#include "animation/AnimationTimeline.h"

#include <optional>

struct AnimationIcons;

// Sets `scrubbing` while the frame marker is pressed.
std::optional<action::timeline::Action> RenderAnimationTimeline(const TimelineRange &, const TimelinePlayback &, const AnimationTimelineView &, const TimelineNavigation &, const AnimationIcons &, bool &scrubbing);

// Arrow keys offset one frame, Shift+arrows jump to a range end, and Ctrl+arrows jump by the navigation delta.
// Escape cancels playback while playing.
// Yields to text input and to keyboard widget navigation.
std::optional<action::timeline::Action> HandleTimelineShortcuts(const TimelinePlayback &);
