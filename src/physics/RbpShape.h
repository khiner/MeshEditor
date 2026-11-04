#pragma once

#include "PhysicsTypes.h"
#include "gpu/Shared.h"

#include <span>

namespace rbp {
struct World;
} // namespace rbp

namespace physics {
// Scale is baked into geometry before placing the collider in its owner's rigid frame.
// A mesh-backed shape takes its mesh's positions by vertex ordinal and, for a triangle mesh, its triangles as ordinal triples.
rbp::Index BuildRbpShape(rbp::World &, const PhysicsShape &, std::span<const vec3> positions, std::span<const uint32_t> triangles, vec3 scale, rbp::Pose local);
} // namespace physics
