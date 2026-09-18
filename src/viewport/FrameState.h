#pragma once

#include "numeric/vec2.h"

#include <cstdint>
#include <optional>

using numeric::vec2;

struct FrameState {
    float DeltaTime{0};
    bool FixedFrameStep{false};
    vec2 DisplayFramebufferScale{1, 1};
    vec2 AccumulatedWrapMouseDelta{0, 0};
    vec2 PreciseWheelDelta{0, 0};
    std::optional<vec2> BoxSelectStart, BoxSelectEnd;
    bool BoxSelectStaged{false};
    bool OverlayControlsHovered{false};
    bool RenderPending{false};
    bool Scrubbing{false};
    bool MotionBlurred{false};
    bool Capturing{false};
    // Recompile every shader on the next event pass.
    bool RecompileShaders{false};
};
