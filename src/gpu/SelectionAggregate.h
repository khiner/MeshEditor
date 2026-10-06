#pragma once

#include "gpu/AABB.h"
#include "gpu/Types.h"

// Aggregate flag bits.
// Selected bits cover selected edges and faces, and the edges incident to selected vertices.
GPU_CONSTANT uint32_t SelectionSelectedSharp = 1u;
GPU_CONSTANT uint32_t SelectionSelectedSmooth = 2u;
GPU_CONSTANT uint32_t SelectionLiveSharp = 4u;
GPU_CONSTANT uint32_t SelectionLiveSmooth = 8u;
GPU_CONSTANT uint32_t SelectionBoundary = 16u; // An edge block holds an edge without an opposite halfedge.

// One canonical 256-element block of a selectable domain.
// Vertex blocks sum selected positions. Vertex and face blocks bound live geometry.
struct SelectionAggregate {
    vec3 PositionSum DEFAULT();
    uint32_t Selected DEFAULT();
    AABB Bounds DEFAULT();
    uint32_t LiveCount DEFAULT();
    uint32_t Flags DEFAULT();
    uint32_t Hidden DEFAULT();
};
static_assert(sizeof(SelectionAggregate) == 52, "SelectionAggregate size");
