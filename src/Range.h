#pragma once

#include <cstdint>

constexpr uint32_t InvalidOffset{~0u};

struct Range {
    uint32_t Offset{0}, Count{0};
};
