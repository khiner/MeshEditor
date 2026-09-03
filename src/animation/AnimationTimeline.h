#pragma once

#include "entt_fwd.h"

// Changes invalidate baked physics frames.
struct TimelineRange {
    int StartFrame{1}, EndFrame{250};
    float Fps{24.f};
};

// Per-tick state that does not invalidate the physics cache.
struct TimelinePlayback {
    int CurrentFrame{1};
    bool Playing{false};
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

// Forces Rebuild on the next tick even when the start frame is cached.
struct PhysicsCacheInvalid {};

// Resets playback to the start frame and invalidates the physics cache.
void JumpToStartFrame(entt::registry &, entt::entity viewport);
