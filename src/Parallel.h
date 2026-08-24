#pragma once

#include <cstdint>
#include <dispatch/dispatch.h>
#include <utility>

// Runs `body(i)` for every i in [0, count), spreading the work over the machine's cores.
// The caller owns ordering: `body` must touch only what belongs to its own index.
template<typename BodyT> void ParallelFor(uint32_t count, BodyT &&body) {
    if (count == 0) return;
    if (count == 1) {
        body(0u);
        return;
    }
    dispatch_apply(count, DISPATCH_APPLY_AUTO, ^(size_t i) { body(uint32_t(i)); });
}
