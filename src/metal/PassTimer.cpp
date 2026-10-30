#include "metal/PassTimer.h"
#include "metal/AutoreleaseScope.h"

#include "metal/MetalCpp.h"

namespace mtl {
PassTimer::~PassTimer() { AutoreleaseScope::Release(CounterHeap); }

std::unique_ptr<PassTimer> PassTimer::Create(const Context &ctx, uint32_t max_passes) {
    const AutoreleaseScope pool;
    const auto descriptor = NS::TransferPtr(MTL4::CounterHeapDescriptor::alloc()->init());
    descriptor->setType(MTL4::CounterHeapTypeTimestamp);
    descriptor->setCount(max_passes * 2);
    NS::Error *error = nullptr;
    auto heap = NS::TransferPtr(ctx.Device->newCounterHeap(descriptor.get(), &error));
    if (!heap) return nullptr;
    const auto frequency = ctx.Device->queryTimestampFrequency();
    if (!frequency) return nullptr;
    return std::unique_ptr<PassTimer>{new PassTimer{std::move(heap), frequency, max_passes}};
}

std::optional<uint32_t> PassTimer::Claim(std::string_view name) {
    if (Names.size() >= MaxPasses) return {};
    Names.emplace_back(name);
    return uint32_t(Names.size() - 1);
}

std::vector<PassTimer::Pass> PassTimer::Resolve() {
    const AutoreleaseScope pool;
    std::vector<Pass> passes;
    if (Names.empty()) return passes;

    const auto *data = CounterHeap->resolveCounterRange(NS::Range::Make(0, Names.size() * 2));
    if (!data) return passes;
    const auto *entries = static_cast<const MTL4::TimestampHeapEntry *>(data->bytes());
    if (!entries || data->length() < Names.size() * 2 * sizeof(MTL4::TimestampHeapEntry)) return passes;

    const auto ms_per_tick = 1e3f / float(TicksPerSecond);
    passes.reserve(Names.size());
    for (size_t i = 0; i < Names.size(); ++i) {
        const auto start = entries[i * 2].timestamp, end = entries[i * 2 + 1].timestamp;
        const auto previous_start = std::exchange(LastTimestamps[i * 2], start);
        const auto previous_end = std::exchange(LastTimestamps[i * 2 + 1], end);
        // An encoder with no executed work can leave either timestamp untouched.
        if (!start || !end || end < start || start == previous_start || end == previous_end) continue;
        passes.emplace_back(Names[i], float(end - start) * ms_per_tick);
    }
    return passes;
}
} // namespace mtl
