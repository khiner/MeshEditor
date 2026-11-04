#pragma once

#include "gpu/ConnectivityRef.h"
#include "gpu/ElementWork.h"

// A record's live halfedges, one thread per slot of its halfedge blocks, and the scratch runs that link its candidate boundary halfedges into loops.
// A candidate has a face and no opposite, its edge is among the sorted Edges or any edge when EdgeCount is zero, and its face has a neighbor.
// Each per-halfedge run holds one word per handle below HalfedgeBound, and a doubled run holds a round's input and output halves.
struct BoundaryLoopsPushConstants {
    ConnectivityRef Connectivity DEFAULT();
    DomainBlocks Blocks DEFAULT(); // Halfedge blocks
    uint32_t HalfedgeBound DEFAULT(), VertexBound DEFAULT();
    uint32_t ScratchSlot DEFAULT(InvalidSlot); // The buffer every run below offsets into
    uint32_t Edges DEFAULT(), EdgeCount DEFAULT();
    uint32_t MaxSides DEFAULT(); // The longest loop kept, or zero for any length
    // Per halfedge: whether it is a candidate, its successor in its sheet, and the candidate claiming it as that successor.
    // Then per halfedge: the candidate winning it as a fallback and the candidate it links to.
    uint32_t Candidate DEFAULT(), Natural DEFAULT(), Claim DEFAULT(), Fallback DEFAULT(), Next DEFAULT();
    uint32_t Outgoing DEFAULT(); // Per vertex: its candidate starts, then per vertex the lowest of them
    uint32_t Labels DEFAULT(), Jumps DEFAULT(), Distances DEFAULT(), Successors DEFAULT(); // Per halfedge, doubled: its loop label and jump, and its distance and successor along the loop, the latter two in the link runs once the labels have converged
    uint32_t Closed DEFAULT(); // Per halfedge: whether its links close into a loop, in the jump half the label rounds leave stale
    uint32_t Lengths DEFAULT(), Flags DEFAULT(); // Per halfedge then a terminator: its loop's length at its representative, and one at each representative, each scanned in place
    uint32_t Records DEFAULT(), Output DEFAULT(); // Per loop: its first pair and its length, then per loop member its origin vertex and itself, in the link runs once the ranks have converged
    uint32_t Parameter DEFAULT(); // A round's buffer parity
};
