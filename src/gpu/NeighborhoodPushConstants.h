#pragma once

#include "gpu/ConnectivityRef.h"
#include "gpu/ElementWork.h"

// A record of a listing over several records, whose elements resolve through its tables and read its face primitives.
struct NeighborhoodRecord {
    ElementTables Tables DEFAULT();
    SlotOffset FacePrimitives DEFAULT();
};

// Gathers of a record's connectivity around listed elements, one thread per listed element.
// A count pass writes each element's item count, the host scans the counts into offsets, and a fill pass writes the items from each offset.
// The per-element passes take a listing over several records.
// A face corner item is (halfedge, vertex, edge, opposite halfedge, opposite face, face sides, other side, its face, its vertex).
// A vertex edge item is (edge, other vertex, face sides, first halfedge, its face), and a vertex corner item is (corner, face, face valence).
// A listed corner item is (vertex, origin, edge, face, next, previous, opposite), and a listed edge item is its first halfedge's (origin, vertex).
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
    uint32_t Positions DEFAULT(); // Per listed vertex or face: its position or normal
    SlotOffset FacePrimitives DEFAULT(); // The record's face primitive indices by handle
    // Per listed element: its record's index into the NeighborhoodRecord run at RecordTable, which replaces Connectivity's tables and FacePrimitives for it.
    // A listing over one record holds InvalidOffset.
    uint32_t Records DEFAULT(InvalidOffset), RecordTable DEFAULT();
};

// A record's largest vertex coordinate magnitude, one thread per slot of its vertex blocks.
// Extent is the word that holds the magnitude's bits, which start at zero.
struct VertexExtentJob {
    ConnectivityRef Connectivity DEFAULT();
    DomainBlocks Blocks DEFAULT(); // Vertex blocks
    SlotOffset Extent DEFAULT();
    uint32_t VertexSlot DEFAULT(InvalidSlot);
};
