#pragma once

#include "Range.h"

// Canonical ranges of a mesh's tetrahedral wireframe geometry, in the tet arenas.
// The geometry is written once when a modal solve loads and read only by the renderer.
struct TetBuffers {
    Range Positions{};
    Range EdgeIndices{};
};
