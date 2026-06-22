#pragma once

#include "Range.h"

#include <vector>

// Authored or evaluated morph target weights for an instance.
// For an animated mesh this is the evaluated weight at the current frame, for a static mesh the authored glTF node weights.
struct MorphWeightState {
    std::vector<float> Weights; // Current weights (CPU), size == target count
};

// GPU allocation for an instance's morph weights in MorphWeightBuffer.
struct MorphWeightGpuRange {
    Range Weights;
};
