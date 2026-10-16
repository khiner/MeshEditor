#pragma once

#include "gpu/Types.h"

// A record's table runs in the member, meshlet, primitive and group table arenas, which resolve its render handles.
// Groups is InvalidOffset for a record without groups.
struct RenderTableRuns {
    uint32_t Members DEFAULT(), Meshlets DEFAULT(), Primitives DEFAULT(), Groups DEFAULT();
};
