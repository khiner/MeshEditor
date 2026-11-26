#pragma once

#include "gpu/ElementWork.h"

// One owner's meshlets to refit, at their arena addresses.
struct MeshletBoundsJob {
    ElementWork MeshletAddresses DEFAULT();
    uint32_t Count DEFAULT();
    uint32_t VertexBase DEFAULT(), LocalTriangleBase DEFAULT(); // The owner's payload range bases
    SlotOffset VertexTable DEFAULT(), CornerTable DEFAULT(); // The owner's table runs, which resolve its cluster vertex references
    uint32_t MeshletSlot DEFAULT(InvalidSlot), MeshletVertexSlot DEFAULT(InvalidSlot), LocalTrianglesSlot DEFAULT(InvalidSlot);
    uint32_t CornerSlot DEFAULT(InvalidSlot), VertexSlot DEFAULT(InvalidSlot);
};

// Each tile of the dispatch refits MeshletBoundsTile meshlets of one job.
GPU_CONSTANT uint32_t MeshletBoundsTile = 32u;
