#pragma once

#include "ProcessEvents.h"
#include "gpu/Transform.h"
#include "state/Entity.h"
#include "viewport/RenderView.h"
#include "viewport/ViewportSchema.h"

using numeric::vec2;

// A mesh operator driven by the mouse, staged again on every change.
// A loop slide's factor is the mouse's travel from StartPx along RailPx in units of its length.
struct MeshOperatorDrag {
    enum class Op : uint8_t { Inset,
                              BevelEdges,
                              BevelVertices,
                              Knife,
                              LoopSlide };
    Op Value{Op::Inset};
    vec2 StartPx{}, CenterPx{};
    float WorldPerPx{1.f};
    vec2 RailPx{}; // The picked edge's half from its center to its positive end in logical pixels
    vec3 Positive{}; // The picked edge's mesh-local direction toward its positive end
    vec3 Center{}; // The picked edge's mesh-local midpoint, where the cut inserts the new loop's reference vertex
    state::Entity MeshEntity{state::Null};
    uint32_t Reference{InvalidOffset}; // The new loop's vertex at Center, found once the cut has cooked
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
    struct OperationDrag {
        state::Entity Node{state::Null};
        Transform Value;
    };
    std::optional<OperationDrag> OperationStart;
    // The acting scene, set from ActingScene before actions apply and replayed from each recorded action.
    state::Entity Scene{state::Null};
    float DeltaTime{0};
    bool FixedFrameStep{false};
    vec2 DisplayFramebufferScale{1, 1};
    vec2 AccumulatedWrapMouseDelta{0, 0};
    vec2 PreciseWheelDelta{0, 0};
    bool PreciseScrolling{false};
    float Magnification{0}; // The trackpad pinch's summed magnification
    std::optional<BoxSelectDrag> BoxDrag;
    std::optional<MeshOperatorDrag> MeshDrag;
    bool OverlayControlsHovered{false};
    bool Scrubbing{false};
    bool Capturing{false};
    bool RecompileShaders{false};
};
