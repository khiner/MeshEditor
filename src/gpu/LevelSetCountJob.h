#pragma once
#include "gpu/NamedAttributeRef.h"
#include "gpu/SlotOffset.h"

// Adds the crossings and the triangles of a lattice's level set to the two words at Counts, one thread per lattice point in tiles of JobTileThreads.
// The point of ordinal v holds its field value at vertex handle v of Values.
struct LevelSetCountJob {
    NamedAttributeRef Values DEFAULT();
    GpuArray<uint32_t, 3> Size DEFAULT();
    float Level DEFAULT();
    uint32_t Points DEFAULT();
    SlotOffset Counts DEFAULT();
};
