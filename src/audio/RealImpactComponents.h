#pragma once

#include "state/Entity.h"
#include <filesystem>
#include <vector>

// Tracks which RealImpact microphone entity is currently providing sample data for this sound object.
struct RealImpactActiveMicrophone {
    state::Entity Entity;
};
// A RealImpact microphone position in the dataset.
struct RealImpactMicrophone {
    uint32_t Index;
};
// Mesh vertex indices for each RealImpact impact point, in impact-index order.
// Used to pair samples (indexed by impact) with mesh vertices on mic swap.
struct RealImpactVertices {
    std::vector<uint32_t> Vertices;
    std::filesystem::path Samples; // Immutable sample file for every microphone.
};
