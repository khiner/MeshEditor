#pragma once

#include <cstdint>
#include <unordered_set>

// Singleton components on the viewport entity.
struct NameRegistry {
    std::unordered_set<std::string> Names;
};
struct ObjectIdCounter {
    uint32_t Next{1};
};
