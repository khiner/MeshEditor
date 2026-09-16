#pragma once

#include "gpu/SlotOffset.h"
#include "gpu/Types.h"

// Defines one mesh's halfedge-connectivity output and scratch layout.
// Output order is outgoing halfedges, opposites, edge-first bits, ranks, and samples.
struct MeshConnectivityJob {
    SlotOffset Corners DEFAULT();
    SlotOffset Connectivity DEFAULT();
    uint32_t VertexCount DEFAULT();
    uint32_t HalfedgeCount DEFAULT();
    uint32_t WordCount DEFAULT();
    // Lower-endpoint buckets with counts, scatter cursors, and halfedge items.
    uint32_t CountsOffset DEFAULT();
    uint32_t CursorsOffset DEFAULT();
    uint32_t ItemsOffset DEFAULT();
    uint32_t BlockOffset DEFAULT();
    uint32_t BlockCount DEFAULT();
    uint32_t PopcountOffset DEFAULT();
    uint32_t WordBlockOffset DEFAULT();
    uint32_t WordBlockCount DEFAULT();
    // Stores total edge count and a non-manifold edge marker.
    uint32_t StateOffset DEFAULT();
};
static_assert(sizeof(MeshConnectivityJob) == 64, "MeshConnectivityJob size");
