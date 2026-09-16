#pragma once

#include "gpu/SlotOffset.h"
#include "gpu/Types.h"

// Defines one normal-derivation item with optional posed positions and push-constant-selected outputs.
struct NormalDeriveEntry {
    uint32_t PosedPositionOffset DEFAULT(InvalidOffset);
    SlotOffset Vertices DEFAULT();
    SlotOffset FaceIndices DEFAULT();
    uint32_t VertexCount DEFAULT();
    // Vertex-fan CSR storage contains VertexCount + 1 offsets followed by FanItemEncoding values.
    uint32_t VertexAdjacencyOffset DEFAULT();
    // Normal-sector CSR storage contains SeamCount + 1 offsets followed by FanItemEncoding values.
    uint32_t SeamFanOffset DEFAULT();
    uint32_t SeamCount DEFAULT();
    // Per-face first-triangle offsets bounded by TriangleCount.
    uint32_t FaceDataOffset DEFAULT();
    uint32_t FaceCount DEFAULT();
    uint32_t TriangleCount DEFAULT();
    uint32_t VertexNormalOffset DEFAULT();
    uint32_t SeamNormalOffset DEFAULT();
    uint32_t FaceNormalOffset DEFAULT();
};
static_assert(sizeof(NormalDeriveEntry) == 60, "NormalDeriveEntry size");
