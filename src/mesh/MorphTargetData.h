#pragma once

#include "numeric/vec3.h"

#include <cstdint>
#include <vector>

// Transient morph target data consumed by MeshStore::CreateMesh, then discarded.
struct MorphTargetData {
    uint32_t TargetCount{0};
    // Deltas packed contiguously: target0[vert0..vertN], target1[vert0..vertN], ...
    // Total size = TargetCount * vertex_count
    std::vector<vec3> PositionDeltas;
    std::vector<vec3> NormalDeltas; // Same layout as PositionDeltas; empty if no normal deltas in file
    std::vector<float> DefaultWeights; // Size = TargetCount, from mesh.weights or all zeros
};
