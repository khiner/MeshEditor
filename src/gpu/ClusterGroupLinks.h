#pragma once
#include "gpu/Types.h"

// Edit-only inverse edges for GroupIndex and RefinedGroup, respectively.
// Both runs are offsets in the owner's group cluster range.
struct ClusterGroupLinks {
    uint32_t MemberOffset DEFAULT(), MemberCount DEFAULT();
    uint32_t ProxyOffset DEFAULT(), ProxyCount DEFAULT();
};
static_assert(sizeof(ClusterGroupLinks) == 16);
