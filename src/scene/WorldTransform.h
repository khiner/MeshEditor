#pragma once

#include "gpu/Transform.h"
#include "state/Entity.h"

// World-space transform of a node without an instance slot that holds it, composed from the local Transform and parent chain.
// An instance's slot in the UMA transform buffer holds its world transform, and an armature part's slot holds its display transform.
struct WorldTransform : Transform {
    using Transform::Transform;
    WorldTransform(const Transform &t) : Transform{t} {}
};

// The node's world transform from its home, or null for a node the settle pass has not composed yet.
// The pointer stays valid until the next instance slot placement.
const Transform *WorldTransformOf(const state::Scene &, state::Entity);
// Writes the node's world transform to its home and publishes Change::WorldTransform.
void SetWorldTransform(state::Scene &, state::Entity, const Transform &);

// Evaluated local pose of an animated node or a bone. Derived.
// When present, the world transform composes from this instead of Transform. Bones carry only a pose.
struct PosedLocal {
    Transform Value;
};
