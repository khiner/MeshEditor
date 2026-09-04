#pragma once

#include <cstdint>

// Singleton components on the viewport entity.
struct ObjectIdCounter {
    uint32_t Next{1};
};
