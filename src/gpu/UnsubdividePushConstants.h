#pragma once

#include "gpu/ConnectivityRef.h"

// Per-vertex runs hold one word per vertex handle, per-face runs one per handle below FaceBound with a terminator, and per-selected runs one per listed vertex.
GPU_CONSTANT uint32_t UnsubdividePolygonWords = 11u; // Corner count, four corner and edge pairs, then the sum and xor of the polygon's vertex multiset hash, whose size is the corner count
GPU_CONSTANT uint32_t UnsubdivideStateNone = 0u;
GPU_CONSTANT uint32_t UnsubdivideStateCandidate = 2u; // In the removed wave, awaiting its decision
GPU_CONSTANT uint32_t UnsubdivideStateRemoved = 3u;

struct UnsubdividePushConstants {
    ConnectivityRef Connectivity DEFAULT();
    uint32_t FaceCount DEFAULT(), FaceBound DEFAULT();
    uint32_t ScratchSlot DEFAULT(InvalidSlot); // The buffer every run below offsets into
    uint32_t Selection DEFAULT(), SelectionCount DEFAULT(); // The selected vertices, ascending
    uint32_t Eligible DEFAULT(); // Per selected vertex: eligible, with bit 1 set for a chain vertex
    uint32_t Tags DEFAULT(); // Per selected vertex: the wave, written by the host from the parity labels
    uint32_t State DEFAULT(); // Per vertex: one of the states above
    uint32_t Chain DEFAULT(); // Per vertex: whether a chain vertex
    uint32_t Polygons DEFAULT(); // Per selected vertex: its fan polygon
    uint32_t Changed DEFAULT(); // One word a decision round sets when a vertex decides
    uint32_t Removed DEFAULT(); // Per selected vertex then a terminator: whether removed, scanned in place
    uint32_t FaceWords DEFAULT(), FaceFlags DEFAULT(); // Per face then a terminator: its record's words and whether it is touched, each scanned in place
    uint32_t Table DEFAULT(), TableMask DEFAULT(); // A key table of polygon keys and check words over a power of two of slots, whose value is the polygon's list position
    uint32_t Duplicate DEFAULT(); // One word set when two polygons share a hash
    uint32_t List DEFAULT(); // The ReplaceFaces list
    uint32_t Parameter DEFAULT(); // The list's touched face count
};
