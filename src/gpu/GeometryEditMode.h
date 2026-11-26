#pragma once

#include "gpu/Types.h"

// Commit leaves positions and marks no vertex, and Refresh marks every candidate.
enum class GeometryEditMode : uint32_t {
    Commit = 0,
    Refresh = 1,
};
