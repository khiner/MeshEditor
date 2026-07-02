#pragma once

#include "numeric/vec2.h"

#include <cstdint>
#include <optional>

// Per-frame scratch state produced and consumed within a single frame.
// A context singleton; passed by reference into Interact/InteractOverlay/DrawOverlay so producers and consumers see the same instance.
struct FrameState {
    float DeltaTime{0}; // Seconds since the last frame (drives playback advance)
    bool FixedFrameStep{false}; // Playback advances exactly one timeline frame per tick (deterministic render mode)
    vec2 DisplayFramebufferScale{1, 1}; // Logical-to-physical pixel scale (DPI)
    vec2 AccumulatedWrapMouseDelta{0, 0};
    vec2 PreciseWheelDelta{0, 0};
    uint32_t ObjectPickEpochTag{255}; // 8-bit epoch encoded in object click keys; wraps with periodic key reset
    std::optional<vec2> BoxSelectStart, BoxSelectEnd;
    bool BoxSelectStaged{false}; // A box-select staged this gesture, so its release emits the terminal action.
    bool OverlayControlsHovered{false};
    bool RenderPending{false}; // GPU render submitted but not yet waited on
};
