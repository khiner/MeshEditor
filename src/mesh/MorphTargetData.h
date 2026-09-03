#pragma once

struct MorphTargetData {
    uint32_t TargetCount{0};
    // Deltas packed target-major: target0[vert0..vertN], target1[vert0..vertN], ...
    std::vector<vec3> PositionDeltas;
    std::vector<vec3> NormalDeltas;
    std::vector<vec3> TangentDeltas;

    std::vector<float> DefaultWeights;
};
