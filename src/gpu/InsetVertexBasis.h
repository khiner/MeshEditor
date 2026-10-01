#pragma once

#include "gpu/Types.h"

// One output vertex of a staged inset.
// The source geometry fixes these vectors for the gesture.
// Only Thickness and Depth change between preview frames.
struct InsetVertexBasis {
    uint32_t Handle DEFAULT(InvalidOffset);
    vec3 Base DEFAULT(), Width DEFAULT(), Depth DEFAULT();
};
static_assert(sizeof(InsetVertexBasis) == 40, "InsetVertexBasis size");
