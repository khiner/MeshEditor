#pragma once
#include "project/store/Pages.h"
namespace state {
struct Allocation {
    store::VersionedVector<uint32_t> Generations, Free;
};
} // namespace state
