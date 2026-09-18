#pragma once

#include "numeric/vec2.h"

#include <cstdint>
#include <optional>

using numeric::vec2;

// A mesh operator sized by the mouse's distance from the selection's screen center, staged again on every change.
struct MeshOperatorDrag {
    enum class Op : uint8_t { Inset,
                              BevelEdges,
                              BevelVertices,
                              Knife };
    Op Value{Op::Inset};
    vec2 StartPx{}, CenterPx{};
    float WorldPerPx{1.f};
    uint32_t Segments{1};
    bool Individual{false};
    float WheelAccum{0.f};
    std::optional<float> Staged;
};

struct FrameState {
    float DeltaTime{0};
    bool FixedFrameStep{false};
    vec2 DisplayFramebufferScale{1, 1};
    vec2 AccumulatedWrapMouseDelta{0, 0};
    vec2 PreciseWheelDelta{0, 0};
    std::optional<vec2> BoxSelectStart, BoxSelectEnd;
    bool BoxSelectStaged{false};
    std::optional<MeshOperatorDrag> MeshDrag;
    bool OverlayControlsHovered{false};
    bool RenderPending{false};
    bool Scrubbing{false};
    bool MotionBlurred{false};
    bool Capturing{false};
    // Recompile every shader on the next event pass.
    bool RecompileShaders{false};
};
