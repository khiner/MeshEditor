#pragma once

#include "entt_fwd.h"

// Tracks which RealImpact microphone entity is currently providing sample data for this sound object.
struct RealImpactActiveMicrophone {
    entt::entity Entity;
};
// A RealImpact microphone position in the dataset.
struct RealImpactMicrophone {
    uint32_t Index;
};
