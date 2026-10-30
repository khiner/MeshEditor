#pragma once

#include "metal/MetalContext.h"

#include <Metal/MTL4Counters.hpp>
#include <optional>
#include <string_view>

namespace mtl {
// Per-command GPU timing from timestamps written at each pass's start and end.
struct PassTimer {
    struct Pass {
        std::string_view Name;
        float Ms;
    };

    ~PassTimer();

    static std::unique_ptr<PassTimer> Create(const Context &, uint32_t max_passes = 64);

    // Returns the start/end timestamp-pair index, or nullopt when full.
    std::optional<uint32_t> Claim(std::string_view name);

    MTL4::CounterHeap *Heap() const { return CounterHeap.get(); }

    // Resolves completed timestamp pairs and omits unwritten or already resolved pairs.
    std::vector<Pass> Resolve();

    void Reset() { Names.clear(); }

private:
    PassTimer(NS::SharedPtr<MTL4::CounterHeap> heap, uint64_t ticks_per_second, uint32_t max_passes)
        : CounterHeap(std::move(heap)), TicksPerSecond(ticks_per_second), MaxPasses(max_passes), LastTimestamps(max_passes * 2, 0u) {}

    NS::SharedPtr<MTL4::CounterHeap> CounterHeap;
    uint64_t TicksPerSecond;
    uint32_t MaxPasses;
    std::vector<std::string_view> Names;
    // Reset only clears labels: unused entries can retain the preceding command's timestamps.
    std::vector<uint64_t> LastTimestamps;
};
} // namespace mtl
