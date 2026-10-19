#pragma once

#include "action/Action.h"

struct TimelineIcons;
struct InterfaceTheme;
struct TimelineTheme;

// Draws `keyframes` (sorted timeline frames) as a summary row under the ruler.
// Sets `scrubbing` while the frame marker is pressed.
std::optional<action::Action> RenderTimeline(const TimelineRange &, const TimelinePlayback &, const TimelineView &, state::Registry &, const Animations &, std::span<const float> keyframes, const TimelineIcons &, const InterfaceTheme &, const TimelineTheme &, bool &scrubbing);
void DrawTimelineStyle(state::Registry &);

// Left/Right offset one frame, Shift+Left/Right jump to a range end, and Ctrl+Left/Right jump by the navigation delta.
// Up/Down jump to the previous/next keyframe.
// Escape cancels playback while playing.
// Yields to text input and to keyboard widget navigation.
std::optional<action::timeline::Action> HandleTimelineShortcuts(const TimelinePlayback &);
