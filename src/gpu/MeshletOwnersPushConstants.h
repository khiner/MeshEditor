#pragma once
#include "gpu/ElementAttributeRef.h"

struct MeshletOwnersPushConstants {
    uint32_t First DEFAULT(), Count DEFAULT();
    uint32_t MeshletSlot DEFAULT(), TriangleIdsSlot DEFAULT(), Topology DEFAULT(), ElementOrigin DEFAULT();
    ElementAttributeRef Owners DEFAULT();
    uint32_t BlockCount DEFAULT(), ErrorSlot DEFAULT();
};
static_assert(sizeof(MeshletOwnersPushConstants) == 40);
