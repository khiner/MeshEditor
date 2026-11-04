#pragma once

#include "gpu/ConnectivityRef.h"

// The soup holds three corners per selected face, each with its corner halfedge, vertex and soup edge, and the soup edges with their user corners.
GPU_CONSTANT uint32_t BeautifyNodeWords = 3u; // Source halfedge, vertex, edge
GPU_CONSTANT uint32_t BeautifyEdgeWords = 5u; // Source halfedge, eligible, two user corners, cost bits
GPU_CONSTANT uint32_t BeautifyFlagCandidate = 1u;
GPU_CONSTANT uint32_t BeautifyFlagSelected = 2u;

struct BeautifyPushConstants {
    ConnectivityRef Connectivity DEFAULT();
    uint32_t VertexSlot DEFAULT(InvalidSlot);
    uint32_t ScratchSlot DEFAULT(InvalidSlot); // The buffer every run below offsets into
    uint32_t Faces DEFAULT();
    uint32_t SelectedEdges DEFAULT(), SelectedEdgeCount DEFAULT(); // Ascending
    uint32_t Nodes DEFAULT(), NodeCount DEFAULT(); // Three per selected face, dead for a face that is no triangle
    uint32_t Owners DEFAULT(); // Per corner then a terminator: whether the corner is the lowest with its edge key, scanned in place into compact edge indices
    uint32_t Users DEFAULT(), Second DEFAULT(); // Per corner: at an owner, the corners sharing its key and the highest of them
    uint32_t SoupOwned DEFAULT(); // Per mesh edge: whether every face along it is a soup triangle
    uint32_t Edges DEFAULT(), EdgeCount DEFAULT();
    uint32_t Table DEFAULT(), TableMask DEFAULT(); // Key table over a power of two of slots, valued by the owning corner while building and the winning edge in a round
    uint32_t Flags DEFAULT(); // Per edge: candidate, then selected
    uint32_t Changed DEFAULT(), Rotations DEFAULT();
};
