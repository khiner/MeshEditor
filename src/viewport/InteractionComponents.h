#pragma once

#include "gpu/Element.h"
#include "gpu/InteractionMode.h"

#include <numbers>
#include <set>

struct Interaction {
    InteractionMode Mode{InteractionMode::Object};
};

struct EditMode {
    Element Value{Element::Vertex};
};

struct EnabledInteractionModes {
    std::set<InteractionMode> Value{InteractionMode::Object, InteractionMode::Edit, InteractionMode::Pose};
};

struct OrbitToActive {
    bool Value{false};
};

struct ShadeSmoothAngle {
    float Value{std::numbers::pi_v<float> / 6.f};
};
