#pragma once

#include "gpu/Types.h"

GPU_CONSTANT uint32_t MeshElementBlockSize = 256u;
GPU_CONSTANT uint32_t MeshElementBlockWords = MeshElementBlockSize / 32u;

// A domain's block index determines its canonical element handles. Membership
// links and live masks contain no geometry. Owner names the independent set ID.
struct MeshElementBlock {
    uint32_t Next DEFAULT(InvalidOffset);
    uint32_t Previous DEFAULT(InvalidOffset);
    uint32_t Owner DEFAULT(InvalidOffset);
    uint32_t Count DEFAULT();
    GpuArray<uint32_t, MeshElementBlockWords> Live DEFAULT();
};
static_assert(sizeof(MeshElementBlock) == 48, "MeshElementBlock size");
