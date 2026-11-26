#pragma once

#include "gpu/Types.h"

// Function-constant indices for the shader that declares them.
enum class MeshVertexConstant : uint32_t {
    NonTriangleTopology = 0,
    // The frame has patterned materials, whose triangles only the pattern pipelines draw.
    SplitPatterns,
};
