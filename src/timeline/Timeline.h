#pragma once

#include "Field.h"
#include "state/Entity.h"
#include "timeline/TimelineSchema.h"

// Start-frame and frame-rate changes invalidate baked physics frames.
struct TimelineRange {
    int StartFrame{1}, EndFrame{250};
    float Fps{24.f};
};

// Per-tick state that does not invalidate the physics cache.
// Reverse plays toward the start frame and wraps to the end frame.
// PlayStartFrame is the frame playback started from, restored by CancelPlay.
struct TimelinePlayback {
    int CurrentFrame{1};
    int PlayStartFrame{1};
    bool Playing{false};
    bool Reverse{false};
};

// Frame-step settings for Jump Time by Delta and the arrow-key frame offsets.
// JumpDelta counts frames, or seconds when JumpInSeconds is set.
// Wrap keeps stepped frames inside the timeline range.
struct TimelineNavigation {
    float JumpDelta{1.f};
    bool JumpInSeconds{false};
    bool Wrap{false};
};
template<> inline constexpr field::SpecTable<TimelineNavigation> Specs<TimelineNavigation>{
    {&TimelineNavigation::JumpDelta, {.Min = 0.1, .Speed = 0.1f, .Widget = FieldWidget::Drag, .Label = "Jump delta", .Help = "Distance jump buttons and Ctrl+Left/Right move the current frame, in frames or seconds"}},
    {&TimelineNavigation::JumpInSeconds, {.Label = "Jump in seconds", .Help = "Measure the jump delta in seconds instead of frames"}},
    {&TimelineNavigation::Wrap, {.Label = "Wrap navigation", .Help = "Wrap frame steps and keyframe jumps around the timeline range"}},
};

struct TimelineView {
    float PixelsPerFrame{4.5f};
    float ViewCenterFrame{125.f};
};

// Fractional playback position advanced by Render.
struct PlaybackFrame {
    float Value{1.f};
};

// Frame used to evaluate the current armature, morph, and node poses.
struct LastEvaluatedFrame {
    int Value{-1};
};

// Resets playback to the start frame and invalidates the physics cache.
void JumpToStartFrame(state::Registry &);
