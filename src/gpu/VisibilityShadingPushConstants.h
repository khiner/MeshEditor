#pragma once

#include "gpu/Types.h"

struct VisibilityShadingPushConstants {
    uint32_t PrimitiveSlot DEFAULT();
    uint32_t InstanceSlot DEFAULT();
    uint32_t InstanceMapSlot DEFAULT();
    uint32_t MeshletSlot DEFAULT();
    uint32_t MeshletTriangleSlot DEFAULT();
    uint32_t MeshletLocalTriangleSlot DEFAULT();
    uint32_t MeshletVertexSlot DEFAULT();
    uint32_t VisibleMeshletSlot DEFAULT();
};
static_assert(sizeof(VisibilityShadingPushConstants) == 32, "VisibilityShadingPushConstants size");
