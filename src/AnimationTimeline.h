#pragma once

#include "entt_fwd.h"

// Timeline configuration. Changes here invalidate baked physics frames.
struct TimelineRange {
    int StartFrame{1}, EndFrame{100};
    float Fps{24.0f};
};

// Per-tick playback state. Mutated every play frame; does not affect physics cache.
struct TimelinePlayback {
    int CurrentFrame{1};
    bool Playing{false};
};

struct AnimationTimelineView {
    float PixelsPerFrame{8.0f};
    float ViewCenterFrame{50.0f}; // Frame at horizontal center of visible scroll region
};

// Smooth float frame position for playback, advanced by Render. Singleton on viewport.
struct PlaybackFrame {
    float Value{1.0f};
};

// Last frame where armature poses were evaluated. Singleton on viewport.
struct LastEvaluatedFrame {
    int Value{-1};
};

// Forces a fresh Rebuild on the next tick even if the start frame is cached.
// Emitted by JumpToStart and LoadGltf; cleared after the cache is cleared.
struct PhysicsCacheInvalid {};

// Reset playback to the timeline's start frame and invalidate the physics cache.
void JumpToStartFrame(entt::registry &, entt::entity viewport);
