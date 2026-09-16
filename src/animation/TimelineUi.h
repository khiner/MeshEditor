#pragma once

#include "action/Timeline.h"
#include "animation/AnimationTimeline.h"

#include <optional>
#include <span>

struct AnimationIcons;

// Draws `keyframes` (sorted timeline frames) as a summary row under the ruler.
// Sets `scrubbing` while the frame marker is pressed.
std::optional<action::timeline::Action> RenderAnimationTimeline(const TimelineRange &, const TimelinePlayback &, const AnimationTimelineView &, const TimelineNavigation &, std::span<const float> keyframes, const AnimationIcons &, bool &scrubbing);

// Left/Right offset one frame, Shift+Left/Right jump to a range end, and Ctrl+Left/Right jump by the navigation delta.
// Up/Down jump to the previous/next keyframe.
// Escape cancels playback while playing.
// Yields to text input and to keyboard widget navigation.
std::optional<action::timeline::Action> HandleTimelineShortcuts(const TimelinePlayback &);
