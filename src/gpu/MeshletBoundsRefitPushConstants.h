#pragma once

#include "gpu/ElementWork.h"

struct MeshletBoundsRefitPushConstants {
    ElementWork MeshletAddresses DEFAULT();
    uint32_t Count DEFAULT();
    uint32_t MeshletSlot DEFAULT(InvalidSlot), MeshletVertexSlot DEFAULT(InvalidSlot), LocalTrianglesSlot DEFAULT(InvalidSlot);
    uint32_t CornerSlot DEFAULT(InvalidSlot), VertexSlot DEFAULT(InvalidSlot);
    uint32_t VertexBase DEFAULT(), LocalTriangleBase DEFAULT(); // The owner's payload range bases
    SlotOffset VertexTable DEFAULT(), CornerTable DEFAULT(); // The owner's table runs, which resolve its cluster vertex references
};
static_assert(sizeof(MeshletBoundsRefitPushConstants) == 64, "MeshletBoundsRefitPushConstants size");
