#pragma once

#include "TransformGizmo.h" // GizmoTransform

// Per-frame scratch state produced and consumed within a single frame.
// A component on the viewport entity; passed by reference into Interact/InteractOverlay/DrawOverlay so producers and consumers see the same instance.
struct FrameState {
    vec2 AccumulatedWrapMouseDelta{0, 0};
    vec2 PreciseWheelDelta{0, 0};
    uint32_t ObjectPickEpochTag{255}; // 8-bit epoch encoded in object click keys; wraps with periodic key reset
    std::optional<vec2> BoxSelectStart, BoxSelectEnd;
    std::optional<GizmoTransform> GizmoRenderTransform; // Set by InteractOverlay, consumed by DrawOverlay
    bool OverlayControlsHovered{false};
    bool RenderPending{false}; // GPU render submitted but not yet waited on
};
