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

// Reset playback to the timeline's start frame and invalidate the physics cache.
void JumpToStartFrame(entt::registry &, entt::entity viewport);
