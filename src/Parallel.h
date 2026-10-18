#pragma once

#include <dispatch/dispatch.h>

// Run body(i) concurrently for i in [0, count).
// Each invocation must access only state assigned to its index.
void ParallelFor(uint32_t count, auto &&body) {
    if (count == 0) return;
    if (count == 1) {
        body(0u);
        return;
    }
    dispatch_apply(count, DISPATCH_APPLY_AUTO, ^(size_t i) { body(uint32_t(i)); });
}
