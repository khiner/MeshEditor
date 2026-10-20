#pragma once

#include "physics/PhysicsSchema.h"
#include "state/DirtySet.h"

// Physics change sets, bound by physics::Init.
struct PhysicsChanges {
    state::DirtySet Input; // Any body, collider, joint, hierarchy, or scene membership input.
    state::DirtySet Transform; // A local transform update, relevant when the entity poses a body, collider, or joint.
    state::DirtySet MaterialDef, CollisionSystemDef, CollisionFilterDef;
    state::DirtySet Scenes; // A scene was created or destroyed, kept until physics::ProcessChanges reads it.
    state::DirtySet Geometry, BodyMesh, ColliderPolicy;
    state::DirtySet Colliders; // A collider shape was created, changed or destroyed.
    state::DirtySet DefinitionUses; // A reference to a physics material, collision system, collision filter, or joint definition changed.
};
