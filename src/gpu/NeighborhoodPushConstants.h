#pragma once

#include "gpu/ConnectivityRef.h"

// Gathers of a record's connectivity around listed elements, one thread per listed element.
// A count pass writes each element's item count, the host scans the counts into offsets, and a fill pass writes the items from each offset.
// A face corner item is (halfedge, vertex, edge, opposite halfedge, opposite face, face sides, other side, its face, its vertex).
// A vertex edge item is (edge, other vertex, face sides, first halfedge, its face), and a vertex corner item is (corner, face, face valence).
// A listed corner item is (vertex, origin, edge, face, next, previous, opposite).
// The nearest pass instead lists candidate vertex blocks per query point, point i's run spanning Counts[i] to Counts[i + 1].
// It writes (handle, squared distance) per point.
struct NeighborhoodPushConstants {
    ConnectivityRef Connectivity DEFAULT();
    uint32_t VertexSlot DEFAULT(InvalidSlot), FaceNormalSlot DEFAULT(InvalidSlot), VertexNormalSlot DEFAULT(InvalidSlot);
    uint32_t VertexBlocksSlot DEFAULT(InvalidSlot);
    uint32_t ScratchSlot DEFAULT(InvalidSlot); // The buffer every run below offsets into
    uint32_t Elements DEFAULT(), ElementCount DEFAULT();
    uint32_t Counts DEFAULT(); // Per listed element: its item count, or its first item once the host has scanned them
    uint32_t Items DEFAULT(); // The items, as many words each as the gather's record holds
    uint32_t Positions DEFAULT(); // Per listed vertex or face: its position or normal, or one word for an extent
    SlotOffset FacePrimitives DEFAULT(); // The record's face primitive indices by handle
};
