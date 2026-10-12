#pragma once

#include "gpu/MeshletPayloads.h"

struct MeshletDrawPushConstants {
    uint32_t PrimitiveSlot DEFAULT();
    uint32_t InstanceSlot DEFAULT();
    uint32_t InstanceMapSlot DEFAULT();
    uint32_t MeshletSlot DEFAULT();
    MeshletPayloadSlots Payloads DEFAULT();
    uint32_t VisibleMeshletSlot DEFAULT();
    uint32_t RouteStateSlot DEFAULT();
    uint32_t Route DEFAULT();
    uint32_t VisibleOffset DEFAULT();
    uint32_t RequiredInstanceFlags DEFAULT();
    uint32_t EditEdgeCorner DEFAULT();
    uint32_t VisibilityTransmission DEFAULT();
    uint32_t EdgeSharpnessSlot DEFAULT(InvalidSlot);
};
static_assert(sizeof(MeshletDrawPushConstants) == 60, "MeshletDrawPushConstants size");
