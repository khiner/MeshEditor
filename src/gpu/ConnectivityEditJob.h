#pragma once

#include "gpu/ElementHandleRange.h"
#include "gpu/ElementWork.h"

// One record's job in the pass that lists its writable vertices, halfedges and faces after an edit.
struct ConnectivityEditJob {
    GpuArray<ElementWork, 3> Before DEFAULT(); // V, H, F
    GpuArray<uint32_t, 3> Counts DEFAULT();
    GpuArray<ElementWork, 2> Replaced DEFAULT(); // H, F
    GpuArray<ElementHandleRange, 3> Emitted DEFAULT();
    GpuArray<ElementWork, 3> After DEFAULT();
    uint32_t ErrorSlot DEFAULT(InvalidSlot);
};
