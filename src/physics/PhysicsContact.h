#pragma once

#include "entt_fwd.h"
#include "numeric/vec3.h"

#include <vector>

// A discrete impact on one rigid body from a new solid contact, one event per body in a qualifying pair.
// All quantities are world space at the impact frame.
struct ContactImpact {
    entt::entity Entity{null_entity}; // Owning entity of the struck body.
    entt::entity Other{null_entity}; // The body that struck it, the acoustic impactor.
    vec3 Point{0}; // Contact point.
    vec3 Direction{0}; // Unit impulse direction into this body.
    float Impulse{0}; // Contact impulse magnitude, kg·m/s.
    float Speed{0}; // Normal approach speed at the contact, m/s.
    float OtherInvMass{0}; // Inverse mass of the other body, kg⁻¹; 0 = immovable.
};

// Registry-context queue: the physics step appends this frame's impacts, the audio system drains it.
struct PhysicsContactImpacts {
    std::vector<ContactImpact> Events;
};
