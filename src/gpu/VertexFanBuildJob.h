#pragma once
#include "gpu/ElementWork.h"
#include "gpu/SlotOffset.h"

// Scratch contains references and allocation metadata, never copied geometry.
// The fans fill the run of HalfedgeCount items at FirstItem from its start, in vertex work order.
struct VertexFanBuildJob {
    ElementWork Vertices DEFAULT(), Halfedges DEFAULT();
    SlotOffset Corners DEFAULT(), Roots DEFAULT();
    uint32_t ItemsSlot DEFAULT(InvalidSlot), FaceOwnersSlot DEFAULT(InvalidSlot), FaceCount DEFAULT();
    uint32_t VertexCount DEFAULT(), HalfedgeCount DEFAULT(), FirstItem DEFAULT();
    uint32_t Metadata DEFAULT(), Keys DEFAULT(), Order DEFAULT(), Temporary DEFAULT();
    uint32_t Histogram DEFAULT(), Totals DEFAULT(), TileData DEFAULT();
    uint32_t VertexKeyPasses DEFAULT(), Fresh DEFAULT();
};
static_assert(sizeof(VertexFanBuildJob) == 108);
struct VertexFanBuildPushConstants {
    uint32_t StorageSlot DEFAULT(InvalidSlot), JobsOffset DEFAULT(), TileMapOffset DEFAULT(), ScratchOffset DEFAULT();
    uint32_t FirstTile DEFAULT(), PassParameter DEFAULT();
};
static_assert(sizeof(VertexFanBuildPushConstants) == 24);
