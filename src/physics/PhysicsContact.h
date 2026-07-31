#pragma once

#include "entt_fwd.h"
#include "numeric/vec3.h"

#include <array>
#include <vector>

// Marks a body whose contacts the step reports in detail, in ContactImpact and SustainedContact.
// A pair with neither body marked skips the contact solve entirely.
struct ReportContacts {};

// A discrete impact on one rigid body from a new solid contact, one event per contact point per body in a qualifying pair,
// each carrying the load its own point took. All quantities are world space at the impact frame.
struct ContactImpact {
    entt::entity Entity{null_entity}; // Owning entity of the struck body.
    entt::entity ColliderEntity{null_entity}; // The collider node of this body that was struck.
    entt::entity Other{null_entity}; // The body that struck it.
    entt::entity OtherColliderEntity{null_entity}; // The collider node of the other body that did the striking.
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
// The two sweep velocities are independent: a box sliding on a fixed floor has zero sweep on the box and full sweep on the floor.
struct SustainedContactSide {
    entt::entity Entity{null_entity}; // The body this side describes.
    // The collider node of this body that is touching. A compound body carries one per sub-shape.
    entt::entity ColliderEntity{null_entity};
    // Velocity of the contact position over this body's own surface, m/s.
    // A full velocity rather than a speed, since the tangential contact force acts along its direction.
    vec3 SweepVelocity{0};
};

// A contact that persists, one per contact manifold.
// A manifold covers every touching point sharing a contact normal, so a box resting on four corners is a single contact.
// A body wedged between two faces is two, each acting along the direction its own face pushes.
// All quantities are world space, and directions are oriented toward Sides[1], so Sides[0] sees their negation.
struct SustainedContact {
    uint64_t Id{0}; // Stable while the contact lasts, and never reused by another.
    std::array<SustainedContactSide, 2> Sides;
    vec3 Point{0}; // Contact position, one point shared by both bodies.
    vec3 Normal{0}; // Unit contact normal, directed into Sides[1].
    // Tangential velocity of Sides[0]'s material point relative to Sides[1]'s, m/s.
    // Friction opposes a body's own motion, so it acts along this for Sides[1] and against it for Sides[0].
    vec3 Slip{0};
    float NormalForce{0}; // N, non-negative.
    float Restitution{0}; // Combined restitution of the pair.
    float Friction{0}; // Combined friction coefficient of the pair.
};

// The contacts touching as of the last step, rebuilt by every step.
// Level-triggered: a contact this stops naming is over.
struct PhysicsSustainedContacts {
    std::vector<SustainedContact> Active;
    // The simulation step this set was collected from, which advances only when the simulation does.
    // Reading the same value twice means the set is unchanged.
    uint64_t Step{0};
};
