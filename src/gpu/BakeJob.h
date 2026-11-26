#pragma once

#include "gpu/ElementWork.h"

// A bake's coverage of a Width by Height texel grid by a record's triangles, in texel space, where UV (0, 0) is the grid's top left and texel (x, y) spans [x, x + 1] by [y, y + 1].
// Counts holds a word per triangle slot of the record's blocks and a terminator.
// Texels and Jumped hold a word per texel: a triangle address plus one, or zero for none.
struct BakeJob {
    DomainBlocks Triangles DEFAULT();
    uint32_t TrianglesSlot DEFAULT(InvalidSlot); // Triangle arena, by address
    SlotOffset Uvs DEFAULT(); // The baked set's corner UVs, by corner handle
    SlotOffset Counts DEFAULT(), Texels DEFAULT(), Jumped DEFAULT();
    uint32_t Width DEFAULT(), Height DEFAULT();
    uint32_t Step DEFAULT(); // The margin pass's jump in texels
};

// Each tile of a bake pass runs BakeTile of a job's threads.
GPU_CONSTANT uint32_t BakeTile = 256u;
