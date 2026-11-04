#pragma once

#include "gpu/ConnectivityRef.h"
#include "gpu/ElementWork.h"

// A record's live halfedges, one thread per slot of its halfedge blocks.
// Flag is the word that marks the record as holding an edge shorter than Distance.
// Junctions is the list its junctions fill instead: a count, then each edge with the vertex on its interior.
struct ShortEdgesPushConstants {
    ConnectivityRef Connectivity DEFAULT();
    DomainBlocks Blocks DEFAULT(); // Halfedge blocks
    SlotOffset Flag DEFAULT();
    SlotOffset Junctions DEFAULT();
    uint32_t Capacity DEFAULT(); // The pairs the junction list holds
    uint32_t VertexSlot DEFAULT(InvalidSlot);
    float Distance DEFAULT();
};
