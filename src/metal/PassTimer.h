#pragma once

#include "metal/MetalContext.h"

#include <Metal/MTLCounters.hpp>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

namespace mtl {
// Per-command-buffer GPU timing. Hardware samples only at encoder stage boundaries.
struct PassTimer {
    struct Pass {
        std::string_view Name;
        float Ms;
    };

    ~PassTimer();

    static std::unique_ptr<PassTimer> Create(const Context &, uint32_t max_passes = 64);

    // Returns the start/end sample-pair index, or nullopt when full.
    std::optional<uint32_t> Claim(std::string_view name);

    MTL::CounterSampleBuffer *Buffer() const { return SampleBuffer.get(); }

    // Resolves completed sample pairs and omits unwritten or already resolved pairs.
    std::vector<Pass> Resolve();

    void Reset() { Names.clear(); }

private:
    PassTimer(NS::SharedPtr<MTL::CounterSampleBuffer> buffer, uint32_t max_passes)
        : SampleBuffer(std::move(buffer)), MaxPasses(max_passes), LastTimestamps(max_passes * 2, 0u) {}

    NS::SharedPtr<MTL::CounterSampleBuffer> SampleBuffer;
    uint32_t MaxPasses;
    std::vector<std::string_view> Names;
    // Reset only clears labels: unused slots can retain the preceding command's samples.
    std::vector<uint64_t> LastTimestamps;
};
} // namespace mtl
