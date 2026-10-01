#pragma once
#include "gpu/Types.h"
#include "gpu/MeshTopologyArenas.h"
// Shared topology-operator slots with per-pass tiles beginning at FirstTile.
struct MeshTopologyPushConstants {
    uint32_t JobsSlot DEFAULT(InvalidSlot);
    uint32_t TileMapSlot DEFAULT(InvalidSlot);
    uint32_t ScratchSlot DEFAULT(InvalidSlot);
    uint32_t FirstTile DEFAULT();
    uint32_t PassParameter DEFAULT(); // Label domain, convergence arguments, radix shift, or reduction level.
    MeshTopologyArenas Source DEFAULT(), Destination DEFAULT();
    uint32_t ListSlot DEFAULT(InvalidSlot);
};
static_assert(sizeof(MeshTopologyPushConstants) == 296, "MeshTopologyPushConstants size");
