#pragma once

#include "Range.h"

#include <vector>

struct MorphWeightState {
    std::vector<float> Weights; // Current evaluated weights (CPU), size == target count
    Range GpuWeightRange; // Allocation in MorphWeightBuffer
};
