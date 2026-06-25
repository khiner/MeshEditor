#pragma once

#include "gpu/Transform.h"

// Distinct component type so an entity can hold both local and world transforms.
struct WorldTransform : Transform {
    using Transform::Transform;
    WorldTransform(const Transform &t) : Transform{t} {}
};
// World-space transform, composed from the local Transform and parent chain.

// Evaluated local pose of an animated node (its Transform stays the authored local).
// When present, the world transform composes from this instead of Transform. Derived.
struct PosedLocal : Transform {
    using Transform::Transform;
    PosedLocal(const Transform &t) : Transform{t} {}
};
