#pragma once
#include "gpu/SlotOffset.h"
#include "gpu/Types.h"
// Defines one mesh's halfedge-connectivity output and scratch layout.
// Output order is outgoing halfedges, opposites, each halfedge's edge, an n-gon mesh's face starts, then each edge's first halfedge.
// The edge list is sized to the halfedge count, and the host trims it to the edge count.
struct MeshConnectivityJob {
    SlotOffset Corners DEFAULT();
    SlotOffset Connectivity DEFAULT();
    uint32_t VertexCount DEFAULT();
    uint32_t HalfedgeCount DEFAULT();
    uint32_t FaceCount DEFAULT();
    // Nonzero when the run stores each face's first halfedge, which a mesh whose faces are not all triangles needs.
    uint32_t FaceStarts DEFAULT();
    uint32_t WordCount DEFAULT();
    // Open-addressed table keyed by endpoint pair, each slot holding the lowest halfedge of its undirected edge.
    uint32_t TableOffset DEFAULT();
    uint32_t TableMask DEFAULT();
    // Each halfedge's table slot during insertion, then the lowest halfedge of its edge.
    uint32_t RepOffset DEFAULT();
    // Per representative, the lowest halfedge running against it on its edge.
    uint32_t PartnerOffset DEFAULT();
    uint32_t PopcountOffset DEFAULT();
    uint32_t WordBlockOffset DEFAULT();
    uint32_t WordBlockCount DEFAULT();
    // Edge-first bits and their ranks per halfedge word, which number the edges.
    uint32_t BitsOffset DEFAULT();
    uint32_t RanksOffset DEFAULT();
    // Each halfedge's predecessor in its face loop, staged for a mesh whose faces are not all triangles.
    uint32_t PrevOffset DEFAULT(InvalidOffset);
    // Receives the edge count from the rank scan.
    uint32_t StateOffset DEFAULT();
};
static_assert(sizeof(MeshConnectivityJob) == 80, "MeshConnectivityJob size");
