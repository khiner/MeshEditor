#pragma once

#include <cstdint>

constexpr uint32_t InvalidOffset{~0u};

struct Range {
    uint32_t Offset{0}, Count{0};
};

constexpr uint32_t OffsetOrInvalid(Range range) { return range.Count > 0 ? range.Offset : InvalidOffset; }
