#pragma once
#include "gpu/Types.h"

// Edit-only inverse edges for GroupIndex and RefinedGroup, respectively.
// Both packed runs live in the shared group-cluster ID arena.
struct ClusterGroupLinks {
    uint32_t MemberOffset DEFAULT(), MemberCount DEFAULT();
    uint32_t ProxyOffset DEFAULT(), ProxyCount DEFAULT();
};
static_assert(sizeof(ClusterGroupLinks) == 16);
