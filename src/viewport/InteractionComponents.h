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

// Interaction modes available for cycling/selection. Singleton on viewport.
// Excite is present only when the scene has any SoundVertices.
struct EnabledInteractionModes {
    std::set<InteractionMode> Value{InteractionMode::Object, InteractionMode::Edit, InteractionMode::Pose};
};

// In Edit/Excite mode, orbit camera to active element on selection change.
struct OrbitToActive {
    bool Value{false};
};

// Dihedral-angle threshold for the shade-smooth-by-angle operator, in radians.
struct ShadeSmoothAngle {
    float Value{std::numbers::pi_v<float> / 6.f};
};
