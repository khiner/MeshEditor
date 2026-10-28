#pragma once

#include "ProcessEvents.h"
#include "state/Entity.h"
#include "viewport/RenderView.h"
#include "viewport/ViewportSchema.h"

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

struct BoxSelectDrag {
    vec2 Start, End;
    bool Additive; // Shift was held at mouse-down.
    std::optional<std::pair<std::pair<vec2, vec2>, RenderView>> Emitted; // The box and view last staged.
};

// Execution inputs sampled at action boundaries, outside the editable action payload.
struct ActionInputs {
    state::Entity Scene{state::Null};
    float DeltaTime{}, PlaybackFrame{};
    int CurrentFrame{1};
    bool FixedFrameStep{};
    EventPass Pass{EventPass::Frame};
    vec2 FramebufferScale{1, 1};
    bool operator==(const ActionInputs &) const = default;
};

struct FrameState {
    // The acting scene, set from ActingScene before actions apply and replayed from each recorded action.
    state::Entity Scene{state::Null};
    float DeltaTime{0};
    bool FixedFrameStep{false};
    vec2 DisplayFramebufferScale{1, 1};
    vec2 AccumulatedWrapMouseDelta{0, 0};
    vec2 PreciseWheelDelta{0, 0};
    bool PreciseScrolling{false}; // The wheel delta is a trackpad's scroll
    float Magnification{0}; // The trackpad pinch's summed magnification
    std::optional<BoxSelectDrag> BoxDrag;
    std::optional<MeshOperatorDrag> MeshDrag;
    bool OverlayControlsHovered{false};
    bool Scrubbing{false};
    bool Capturing{false};
    // Recompile every shader on the next event pass.
    bool RecompileShaders{false};
};
