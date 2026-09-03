#pragma once

#include <cstdint>
#include <dispatch/dispatch.h>
#include <utility>

// Run `body(i)` concurrently for every i in [0, count).
// Each invocation must access only state assigned to its index.
template<typename BodyT> void ParallelFor(uint32_t count, BodyT &&body) {
    if (count == 0) return;
    if (count == 1) {
        body(0u);
        return;
    }
    dispatch_apply(count, DISPATCH_APPLY_AUTO, ^(size_t i) { body(uint32_t(i)); });
}
