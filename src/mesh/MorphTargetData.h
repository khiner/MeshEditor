#pragma once

#include "numeric/vec3.h"

#include <cstdint>
#include <vector>

// Transient morph target data consumed by MeshStore::CreateMesh, then discarded.
struct MorphTargetData {
    uint32_t TargetCount{0};
    // Deltas packed target-major: target0[vert0..vertN], target1[vert0..vertN], ...
    std::vector<vec3> PositionDeltas;
    std::vector<vec3> NormalDeltas; // empty if source had no normal deltas
    std::vector<vec3> TangentDeltas; // vec3 per spec - handedness isn't displaced

    std::vector<float> DefaultWeights; // size TargetCount; from mesh.weights or zeros
};
