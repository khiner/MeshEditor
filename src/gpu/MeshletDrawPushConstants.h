#pragma once

#include "gpu/Types.h"

struct MeshletDrawPushConstants {
    uint32_t PrimitiveSlot DEFAULT();
    uint32_t InstanceSlot DEFAULT();
    uint32_t InstanceMapSlot DEFAULT();
    uint32_t MeshletSlot DEFAULT();
    uint32_t MeshletTriangleSlot DEFAULT();
    uint32_t MeshletVertexSlot DEFAULT();
    uint32_t MeshletLocalTriangleSlot DEFAULT();
    uint32_t MeshletEditEdgeSlot DEFAULT();
    uint32_t VisibleMeshletSlot DEFAULT();
    uint32_t RouteStateSlot DEFAULT();
    uint32_t Route DEFAULT();
    uint32_t VisibleOffset DEFAULT();
    uint32_t RequiredInstanceFlags DEFAULT();
    uint32_t InstanceFilter DEFAULT(InvalidOffset);
    uint32_t EditEdgeCorner DEFAULT();
    uint32_t VisibilityTransmission DEFAULT();
    uint32_t EdgeSharpnessSlot DEFAULT(InvalidSlot);
};
static_assert(sizeof(MeshletDrawPushConstants) == 68, "MeshletDrawPushConstants size");
