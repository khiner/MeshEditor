#pragma once

#include "gpu/Types.h"
#include "gpu/ElementAttributeRef.h"

struct MeshletBuildPushConstants {
    uint32_t JobsSlot DEFAULT();
    uint32_t ScratchSlot DEFAULT();
    uint32_t TilesSlot DEFAULT(), FirstTile DEFAULT();
    uint32_t PassParameter DEFAULT();
    uint32_t VertexRefsSlot DEFAULT();
    uint32_t TriangleIdsSlot DEFAULT();
    uint32_t LocalTrianglesSlot DEFAULT();
    uint32_t MeshletsSlot DEFAULT();
    uint32_t PrimitivesSlot DEFAULT();
    uint32_t NodesSlot DEFAULT();
    uint32_t PrimitiveRoutesSlot DEFAULT();
    uint32_t LodLeavesSlot DEFAULT(), LodParentsSlot DEFAULT();
    ElementAttributeRef CornerSectors DEFAULT();
    uint32_t FaceSharpnessSlot DEFAULT();
};
static_assert(sizeof(MeshletBuildPushConstants) == 68);
