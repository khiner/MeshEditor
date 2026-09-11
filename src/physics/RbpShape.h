#pragma once

#include "PhysicsTypes.h"
#include "gpu/Shared.h"

namespace rbp {
struct World;
}

struct Mesh;

namespace physics {
// Scale is baked into geometry before placing the collider in its owner's rigid frame.
rbp::Index BuildRbpShape(rbp::World &, const PhysicsShape &, const Mesh *, vec3 scale, rbp::Pose local);
} // namespace physics
