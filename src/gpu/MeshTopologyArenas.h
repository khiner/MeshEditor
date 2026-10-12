#pragma once
#include "gpu/ElementLayers.h"

// Geometry bindings for one version of canonical topology data.
struct MeshTopologyArenas {
    uint32_t VertexSlot DEFAULT(InvalidSlot);
    uint32_t CornerSlot DEFAULT(InvalidSlot);
    uint32_t FaceTriangleStartSlot DEFAULT(InvalidSlot);
    uint32_t TriangleSlot DEFAULT(InvalidSlot);
    uint32_t EdgeSharpnessSlot DEFAULT(InvalidSlot);
    uint32_t FaceSharpnessSlot DEFAULT(InvalidSlot);
    uint32_t BaseVertexNormalSlot DEFAULT(InvalidSlot);
    uint32_t BaseFaceNormalSlot DEFAULT(InvalidSlot);
    uint32_t VertexHiddenSlot DEFAULT(InvalidSlot), EdgeHiddenSlot DEFAULT(InvalidSlot), FaceHiddenSlot DEFAULT(InvalidSlot);
    ElementLayers Layers DEFAULT(); // The layers of the record this version belongs to
};
static_assert(sizeof(MeshTopologyArenas) == 156, "MeshTopologyArenas size");
