#pragma once

#include "gpu/CornerClassificationPushConstants.h"
#include "mesh/TiledJobBatch.h"

// The corner classification of a batch of meshes, with one job per mesh that has vertices to classify.
// Domain zero holds the jobs' vertex tiles, and each target names its job's mesh, its incoming corner bound and whether its result covers the whole mesh.
struct CornerClassBatch {
    struct Target {
        uint32_t StoreId, Incoming;
        bool Complete;
    };
    TiledJobBatch<CornerClassificationJob, 1> Jobs;
    std::vector<Target> Targets{};
};
