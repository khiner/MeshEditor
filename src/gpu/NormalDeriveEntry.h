#pragma once

#include "gpu/ConnectivityRef.h"
#include "gpu/ElementWork.h"
#include "gpu/SlotOffset.h"
#include "gpu/Types.h"

// Defines one normal-derivation item with optional posed positions and push-constant-selected outputs.
struct NormalDeriveEntry {
    uint32_t PositionNamespace DEFAULT(InvalidOffset);
    // Full-weight morph probe reads base plus delta directly, without a position copy.
    SlotOffset Morph DEFAULT();
    uint32_t MorphTargetIndex DEFAULT(), MorphTargetCount DEFAULT();
    uint32_t VerticesSlot DEFAULT(InvalidSlot);
    uint32_t CornersSlot DEFAULT(InvalidSlot);
    uint32_t VertexCount DEFAULT();
    uint32_t VertexBlocksSlot DEFAULT();
    ConnectivityRef Connectivity DEFAULT();
    uint32_t FaceCount DEFAULT();
    uint32_t FaceBlocksSlot DEFAULT();
    SlotOffset CornerSectors DEFAULT(), NormalSectors DEFAULT(); // The record's sector layers, without a slot while it holds none
    uint32_t VertexNormalNamespace DEFAULT(InvalidOffset);
    uint32_t SectorNamespace DEFAULT(InvalidOffset);
    uint32_t FaceNormalNamespace DEFAULT(InvalidOffset);
    // Handles with compact dispatch counts.
    // Absent work uses tiles of live blocks.
    ElementWork VerticesWork DEFAULT(), FacesWork DEFAULT();
    uint32_t VertexWorkCount DEFAULT(), FaceWorkCount DEFAULT();
};
