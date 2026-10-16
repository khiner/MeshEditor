#pragma once

#include "gpu/VisibleMeshlet.h"

// A cull entry's routes and the drawn cluster its classification resolved, which compaction copies to each route.
struct MeshletClassification {
    uint32_t Routes DEFAULT();
    VisibleMeshlet Entry DEFAULT();
};
