#pragma once

#include "gpu/Types.h"

enum class EditSharpnessOperation : uint32_t {
    SetAllFaces = 0,
    SmoothAll = 1,
    SmoothByAngle = 2,
    SetSelectedFaces = 3,
    SetSelectedEdges = 4,
    SetVertexEdges = 5,
};
