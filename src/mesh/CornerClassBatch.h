#pragma once

#include "gpu/CornerClassificationPushConstants.h"
#include "mesh/TiledJobBatch.h"

// The corner classification of a batch of meshes, with one job per mesh that has vertices to classify.
// Domain zero stores vertex tiles.
// Targets identify the mesh, incoming-corner bound, and whole-mesh coverage.
struct CornerClassBatch {
    struct Target {
        uint32_t StoreId, Incoming;
        bool Complete;
    };
    TiledJobBatch<CornerClassificationJob, 1> Jobs;
    std::vector<Target> Targets{};
};
