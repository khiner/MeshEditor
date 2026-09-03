#pragma once

#include "Range.h"

#include <vector>

// Stores evaluated animation weights or authored static glTF node weights.
struct MorphWeightState {
    std::vector<float> Weights; // One CPU value per morph target.
};

struct MorphWeightGpuRange {
    Range Weights;
};
