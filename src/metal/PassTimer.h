#pragma once

#include "metal/MetalContext.h"

#include <string_view>
#include <vector>

namespace mtl {
// Per-command-buffer GPU timing. Hardware samples only at encoder stage boundaries.
struct PassTimer {
    struct Pass {
        std::string_view Name;
        float Ms;
    };

    static std::unique_ptr<PassTimer> Create(const Context &, uint32_t max_passes = 64);

    // Returns the start/end sample-pair index, or nothing when full.
    std::optional<uint32_t> Claim(std::string_view name);

    MTL::CounterSampleBuffer *Buffer() const { return *SampleBuffer; }

    // Call after completion; unwritten samples are omitted.
    std::vector<Pass> Resolve();

    void Reset() { Names.clear(); }

private:
    PassTimer(Owned<MTL::CounterSampleBuffer> buffer, uint32_t max_passes)
        : SampleBuffer(std::move(buffer)), MaxPasses(max_passes) {}

    Owned<MTL::CounterSampleBuffer> SampleBuffer;
    uint32_t MaxPasses;
    std::vector<std::string_view> Names;
};
} // namespace mtl
