// Collider wireframe generation for viewport overlay.
// Canonical unit meshes shared across all instances (Blender-style instancing).
// No Jolt includes — pure geometry.

#pragma once

#include "numeric/vec3.h"

#include <cstdint>
#include <vector>

namespace physics_debug {
struct WireframeMesh {
    std::vector<vec3> Positions;
    std::vector<uint32_t> EdgeIndices; // Line segment pairs
};

// Canonical unit wireframe meshes (generated once, shared via instancing).
WireframeMesh UnitBox(); // [-0.5, 0.5]^3
WireframeMesh UnitSphere(); // Radius 0.5, 3 great circles
WireframeMesh UnitCapsuleCap(); // Hemisphere wireframe, radius 0.5, cap in +Y
WireframeMesh UnitCircle(); // Radius 0.5, in XZ plane (Y-up axis)
WireframeMesh UnitLine(); // Single segment from (0, +0.5, 0) to (0, -0.5, 0)
} // namespace physics_debug
