#pragma once

#include "gpu/SlotOffset.h"
#include "gpu/Types.h"
#include "gpu/VertexAdjacencyKind.h"

// Defines one mesh's vertex-adjacency CSR build and its count/scatter scratch.
struct VertexAdjacencyJob {
    SlotOffset Corners DEFAULT();
    // The mesh's connectivity run, read for face starts and edge tables.
    SlotOffset Connectivity DEFAULT();
    uint32_t VertexCount DEFAULT();
    uint32_t HalfedgeCount DEFAULT();
    uint32_t FaceCount DEFAULT();
    uint32_t FaceStarts DEFAULT();
    VertexAdjacencyKind Kind DEFAULT();
    // CSR storage contains VertexCount + 1 item offsets followed by items.
    uint32_t CsrOffset DEFAULT();
    uint32_t CountsOffset DEFAULT();
    uint32_t BlockOffset DEFAULT();
    uint32_t BlockCount DEFAULT();
};
static_assert(sizeof(VertexAdjacencyJob) == 52, "VertexAdjacencyJob size");
