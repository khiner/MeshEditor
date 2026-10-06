#pragma once
#include "gpu/MeshTopologyArenas.h"
#include "gpu/Types.h"
// Shared topology-operator slots with per-pass tiles beginning at FirstTile.
// The jobs, tile map and scratch are word offsets into the batch's storage slot.
struct MeshTopologyPushConstants {
    uint32_t StorageSlot DEFAULT(InvalidSlot);
    uint32_t JobsOffset DEFAULT(), TileMapOffset DEFAULT(), ScratchOffset DEFAULT();
    uint32_t FirstTile DEFAULT();
    uint32_t PassParameter DEFAULT(); // Label domain, convergence arguments, radix shift, or reduction level.
    MeshTopologyArenas Source DEFAULT(), Destination DEFAULT();
    uint32_t ListSlot DEFAULT(InvalidSlot);
};
static_assert(sizeof(MeshTopologyPushConstants) == 340, "MeshTopologyPushConstants size");
