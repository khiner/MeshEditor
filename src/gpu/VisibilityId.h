#pragma once

#include "gpu/Types.h"

// Packs the local triangle and visible-list meshlet index.
enum class VisibilityId : uint32_t {
    TriangleBits = 6,
    IndexBits = 25,
};
