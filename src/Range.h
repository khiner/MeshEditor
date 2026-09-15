#pragma once

#include "gpu/Types.h"

struct Range {
    uint32_t Offset{0}, Count{0};
};

constexpr uint32_t OffsetOrInvalid(Range range) { return range.Count > 0 ? range.Offset : InvalidOffset; }
