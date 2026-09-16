#pragma once

#include "state/Entity.h"

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

struct AnimationTimelineView {
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
void JumpToStartFrame(state::Scene &, state::Entity viewport);
