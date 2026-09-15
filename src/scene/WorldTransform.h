#pragma once

#include "gpu/Transform.h"

struct WorldTransform : Transform {
    using Transform::Transform;
    WorldTransform(const Transform &t) : Transform{t} {}
};
// World-space transform, composed from the local Transform and parent chain.

// Evaluated local pose of an animated node or a bone. Derived.
// When present, the world transform composes from this instead of Transform. Bones carry only a pose.
struct PosedLocal {
    Transform Value;
};
