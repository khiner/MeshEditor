#pragma once

#include "gpu/ConnectivityRef.h"
#include "gpu/ElementWork.h"

struct NormalDeriveEntry {
    uint32_t PositionNamespace DEFAULT(InvalidOffset);
    SlotOffset Morph DEFAULT();
    uint32_t MorphTargetIndex DEFAULT(), MorphTargetCount DEFAULT();
    uint32_t VerticesSlot DEFAULT(InvalidSlot);
    uint32_t VertexBlocksSlot DEFAULT();
    ConnectivityRef Connectivity DEFAULT();
    uint32_t FaceBlocksSlot DEFAULT();
    SlotOffset CornerSectors DEFAULT(), NormalSectors DEFAULT();
    uint32_t VertexNormalNamespace DEFAULT(InvalidOffset);
    uint32_t SectorNamespace DEFAULT(InvalidOffset);
    uint32_t FaceNormalNamespace DEFAULT(InvalidOffset);
    // Handles with compact dispatch counts.
    // Absent work uses tiles of live blocks.
    ElementWork VerticesWork DEFAULT(), FacesWork DEFAULT();
    uint32_t VertexWorkCount DEFAULT(), FaceWorkCount DEFAULT();
};
