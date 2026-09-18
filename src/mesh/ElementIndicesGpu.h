#pragma once

#include "SlottedRange.h"

#include <cstdint>
#include <span>

namespace state {
struct Scene;
}

// One face mesh's index streams to write, each empty when the mesh does not need it.
struct ElementIndicesWork {
    uint32_t StoreId;
    // Two words per edge, in the index buffer the draws read.
    SlottedRange Endpoints;
    // Three words per fan triangle, in the index buffer the draws read.
    SlottedRange Triangles;
};

// Writes each mesh's edge endpoints and fan triangles on the GPU from its connectivity.
void WriteElementIndicesNow(state::Scene &, std::span<const ElementIndicesWork>);
