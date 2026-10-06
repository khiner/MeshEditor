#pragma once

#include "gpu/ConnectivityRef.h"
#include "gpu/ElementAttributeRef.h"
#include "gpu/ElementWork.h"
#include "gpu/SlotOffset.h"
#include "gpu/Types.h"

// Defines one normal-derivation item with optional posed positions and push-constant-selected outputs.
struct NormalDeriveEntry {
    uint32_t PositionNamespace DEFAULT(InvalidOffset);
    // Full-weight morph probe reads base plus delta directly, without a position copy.
    ElementAttributeRef Morph DEFAULT();
    uint32_t MorphTargetIndex DEFAULT();
    SlotOffset Vertices DEFAULT();
    SlotOffset Corners DEFAULT();
    uint32_t VertexCount DEFAULT();
    uint32_t VertexBlocksSlot DEFAULT();
    ConnectivityRef Connectivity DEFAULT();
    // Face-data arena base, used for derived triangle ownership.
    uint32_t FaceDataOffset DEFAULT();
    uint32_t FaceCount DEFAULT();
    uint32_t FaceBlocksSlot DEFAULT();
    uint32_t HasSectors DEFAULT(1u);
    uint32_t VertexNormalNamespace DEFAULT(InvalidOffset);
    uint32_t SectorNamespace DEFAULT(InvalidOffset);
    uint32_t FaceNormalNamespace DEFAULT(InvalidOffset);
    // Canonical handles with compact dispatch counts.
    // Absent work uses live-block tiles.
    ElementWork VerticesWork DEFAULT(), FacesWork DEFAULT();
    uint32_t VertexWorkCount DEFAULT(), FaceWorkCount DEFAULT();
};
static_assert(sizeof(NormalDeriveEntry) == 168, "NormalDeriveEntry size");
