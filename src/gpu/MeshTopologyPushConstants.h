#pragma once
#include "gpu/MeshTopologyArenas.h"
// The jobs, tile map and scratch are word offsets into the batch's storage slot.
struct MeshTopologyPushConstants {
    uint32_t StorageSlot DEFAULT(InvalidSlot);
    uint32_t JobsOffset DEFAULT(), TileMapOffset DEFAULT(), ScratchOffset DEFAULT();
    uint32_t FirstTile DEFAULT();
    uint32_t PassParameter DEFAULT(); // Label domain, convergence arguments, radix shift, reduction level, or a kernel's stage selector
    MeshTopologyArenas Source DEFAULT(), Destination DEFAULT();
    uint32_t ListSlot DEFAULT(InvalidSlot);
};
