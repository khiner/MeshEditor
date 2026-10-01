#pragma once

#include "gpu/ElementWork.h"

struct MeshletBoundsRefitPushConstants {
    ElementWork Work DEFAULT();
    uint32_t Count DEFAULT();
    uint32_t MeshletSlot DEFAULT(InvalidSlot), MeshletVertexSlot DEFAULT(InvalidSlot), LocalTrianglesSlot DEFAULT(InvalidSlot);
    uint32_t CornerSlot DEFAULT(InvalidSlot), VertexSlot DEFAULT(InvalidSlot);
    uint32_t VertexOffset DEFAULT();
};
static_assert(sizeof(MeshletBoundsRefitPushConstants) == 44, "MeshletBoundsRefitPushConstants size");
