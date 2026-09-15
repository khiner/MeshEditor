#pragma once

#include "gpu/Types.h"

enum class WireCoverage : uint32_t {
    Base = 0,
    Incidental = 1,
    Selected = 2,
    Active = 3,
};
