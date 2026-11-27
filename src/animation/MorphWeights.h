#pragma once

#include "Range.h"
#include "animation/AnimationSchema.h"

// A mesh object's morph weights, one float per morph target in the UMA morph weight buffer.
struct MorphWeightRange {
    Range Weights;
};
