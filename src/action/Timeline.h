#pragma once

#include "timeline/Timeline.h"

namespace action::timeline {
// Enter presentation view (material-preview shading, overlays off) without starting playback.
struct EnterPresentation {};

// Frame pins CurrentFrame on apply, so a recorded stop replays to the same frame.
// Reverse sets the direction when playback starts and is ignored when it stops.
struct TogglePlay {
    int Frame;
    bool Reverse{false};
};
// Stops playback and returns to the frame it started from.
struct CancelPlay {};
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
// Moves the current frame by Delta frames.
struct OffsetFrame {
    int Delta;
};
// Moves the current frame by the TimelineNavigation delta.
struct JumpTime {
    bool Backward;
};
// Moves to the nearest selected-object keyframe after or before the current frame.
struct JumpKeyframe {
    bool Next;
};
struct SetView {
    float PixelsPerFrame, ViewCenterFrame;
};

using Action = std::variant<TogglePlay, CancelPlay, SetFrame, SetStartFrame, SetEndFrame, JumpToStart, JumpToEnd, OffsetFrame, JumpTime, JumpKeyframe, SetView, EnterPresentation>;

void Apply(state::Registry &, state::Entity viewport, const Action &);
} // namespace action::timeline
