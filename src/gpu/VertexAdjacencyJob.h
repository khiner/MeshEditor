#pragma once

#include "gpu/Types.h"
#include "gpu/SlotOffset.h"
#include "gpu/VertexAdjacencyKind.h"

// Defines one mesh's vertex-adjacency CSR build and its count/scatter scratch.
// Edge jobs use edge-first bit ranks to derive edge indices.
struct VertexAdjacencyJob {
    SlotOffset Corners DEFAULT();
    uint32_t VertexCount DEFAULT();
    uint32_t HalfedgeCount DEFAULT();
    VertexAdjacencyKind Kind DEFAULT();
    // CSR storage contains VertexCount + 1 item offsets followed by items.
    uint32_t CsrOffset DEFAULT();
    uint32_t CountsOffset DEFAULT();
    uint32_t BlockOffset DEFAULT();
    uint32_t BlockCount DEFAULT();
    uint32_t EdgeFirstBitsOffset DEFAULT(InvalidOffset);
    uint32_t EdgeFirstRanksOffset DEFAULT(InvalidOffset);
};
static_assert(sizeof(VertexAdjacencyJob) == 44, "VertexAdjacencyJob size");
