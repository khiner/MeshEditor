#pragma once
#include "gpu/ElementTables.h"
#include "gpu/ElementWork.h"

// Scratch contains references and allocation metadata, never copied geometry.
// The fans fill the run of HalfedgeCount items at FirstItem from its start, in vertex work order.
// The record's tables resolve its vertex and corner handles.
struct VertexFanBuildJob {
    ElementWork Vertices DEFAULT(), Halfedges DEFAULT();
    ElementTables Tables DEFAULT();
    uint32_t CornersSlot DEFAULT(InvalidSlot), RootsSlot DEFAULT(InvalidSlot);
    uint32_t ItemsSlot DEFAULT(InvalidSlot), FaceOwnersSlot DEFAULT(InvalidSlot), FaceCount DEFAULT();
    uint32_t VertexCount DEFAULT(), HalfedgeCount DEFAULT(), FirstItem DEFAULT();
    uint32_t Metadata DEFAULT(), Keys DEFAULT(), Order DEFAULT(), Temporary DEFAULT();
    uint32_t Histogram DEFAULT(), Totals DEFAULT(), TileData DEFAULT();
    uint32_t VertexKeyPasses DEFAULT(), Fresh DEFAULT();
};
struct VertexFanBuildPushConstants {
    uint32_t StorageSlot DEFAULT(InvalidSlot), JobsOffset DEFAULT(), TileMapOffset DEFAULT(), ScratchOffset DEFAULT();
    uint32_t FirstTile DEFAULT(), PassParameter DEFAULT();
};
