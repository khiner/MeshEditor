#pragma once

#include "entt_fwd.h"
#include "numeric/vec3.h"

#include <array>
#include <vector>

// Enables ContactImpact and SustainedContact reporting for a body.
struct ReportContacts {};

// Describes one body's world-space response at one point of a new solid contact.
// Impulse excludes static support and is sampled one substep after impact.
struct ContactImpact {
    entt::entity Entity{null_entity}; // Owning entity of the struck body.
    entt::entity ColliderEntity{null_entity}; // The collider node of this body that was struck.
    entt::entity Other{null_entity}; // The body that struck it.
    entt::entity OtherColliderEntity{null_entity}; // The collider node of the other body that did the striking.
    vec3 Point{0}; // Contact point.
    // Load-weighted manifold center that preserves resultant force and moment.
    vec3 ResultantPoint{0};
    vec3 Direction{0}; // Unit impulse direction into this body.
    float Impulse{0}; // Contact impulse magnitude, kg·m/s.
    float Speed{0}; // Normal approach speed the manifold's strike arrested, derived from its impulse, m/s.
    float OtherInvMass{0}; // Inverse mass of the other body, kg⁻¹; 0 = immovable.
    // Manifold polygon area in square meters, or zero for point and edge contacts.
    float NominalArea{0};
};

// Physics appends impacts and audio drains them once per frame.
struct PhysicsContactImpacts {
    std::vector<ContactImpact> Events;
};

// Stores one body's state in a persistent contact as specified by KHR_audio_rigid_bodies.
// The two sweep velocities are independent: a box sliding on a fixed floor has zero sweep on the box and full sweep on the floor.
struct SustainedContactSide {
    entt::entity Entity{null_entity}; // The body this side describes.
    // Collider node for this sub-shape.
    entt::entity ColliderEntity{null_entity};
    // Contact-point velocity over this surface in meters per second.
    vec3 SweepVelocity{0};
};

// A contact that persists, one per contact manifold.
// A manifold contains all contact points sharing a normal.
// Directions use world space and point toward Sides[1].
struct SustainedContact {
    uint64_t Id{0}; // Stable while the contact lasts, and never reused by another.
    std::array<SustainedContactSide, 2> Sides;
    vec3 Point{0}; // Contact position, one point shared by both bodies.
    vec3 Normal{0}; // Unit contact normal, directed into Sides[1].
    // Tangential velocity of Sides[0] relative to Sides[1] in meters per second.
    vec3 Slip{0};
    float NormalForce{0}; // N, non-negative.
    // World-space tangential force applied to Sides[1] in newtons.
    vec3 FrictionForce{0};
    // Manifold polygon area in square meters, or zero for point and edge contacts.
    float NominalArea{0};
    // Manifold extent along the slide in meters.
    float NominalExtent{0};
    float Restitution{0}; // Combined restitution of the pair.
    float Friction{0}; // Combined friction coefficient of the pair.
};

// Contains the contacts from the latest simulation step.
struct PhysicsSustainedContacts {
    std::vector<SustainedContact> Active;
    // Simulation step that produced Active.
    uint64_t Step{0};
};
