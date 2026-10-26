#pragma once

#include "gpu/Types.h"

// Preview poses each candidate vertex at its canonical position, Commit leaves positions and marks no vertex, and Refresh marks every candidate.
enum class GeometryEditMode : uint32_t {
    Preview = 0,
    Commit = 1,
    Refresh = 2,
};
