#pragma once

#include "entt_fwd.h"
#include "numeric/vec3.h"

#include <array>
#include <vector>

// Marks a body whose contacts the step reports in detail, in ContactImpact and SustainedContact.
// A pair with neither body marked skips the contact solve in the collision callback, so a scene of silent bodies costs nothing to leave resting.
struct ReportContacts {};

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

// What one body of a persisting contact answers for on its own (KHR_audio_rigid_bodies contact state).
// The two sweep speeds are independent: a box sliding on a fixed floor has zero sweep on the box and full sweep on the floor.
struct SustainedContactSide {
    entt::entity Entity{null_entity}; // The body this side describes.
    float SweepSpeed{0}; // Rate the contact travels over this body's own surface, m/s.
};

// A contact that persists, reported every step it lasts, one per entity pair, so a box resting on four corners is a single contact.
// All quantities are world space, as in ContactImpact, and the directions are oriented toward Sides[1], so Sides[0] sees their negation.
struct SustainedContact {
    uint64_t Id{0}; // Stable while the contact lasts, so a consumer tracks it across steps.
    std::array<SustainedContactSide, 2> Sides;
    vec3 Point{0}; // Contact position, one point shared by both bodies.
    vec3 Normal{0}; // Unit contact normal, directed into Sides[1].
    vec3 Slip{0}; // Tangential velocity of Sides[0]'s material point relative to Sides[1]'s, m/s. Friction on a body opposes its own motion, so it acts along this for Sides[1] and against it for Sides[0].
    float NormalForce{0}; // N, non-negative.
    float Restitution{0}; // Combined restitution of the pair.
};

// Registry-context list of the contacts touching as of the last step, rebuilt by every step.
// Level-triggered: a consumer tracks what it opened and ends whatever this stops naming.
struct PhysicsSustainedContacts {
    std::vector<SustainedContact> Active;
};
