#include "metal/PassTimer.h"

namespace mtl {
namespace {
MTL::CounterSet *TimestampCounterSet(const Context &ctx) {
    auto *sets = ctx.Device->counterSets();
    for (NS::UInteger i = 0; sets && i < sets->count(); ++i) {
        auto *set = static_cast<MTL::CounterSet *>(sets->object(i));
        if (set->name()->isEqualToString(MTL::CommonCounterSetTimestamp)) return set;
    }
    return nullptr;
}
} // namespace

std::unique_ptr<PassTimer> PassTimer::Create(const Context &ctx, uint32_t max_passes) {
    if (!ctx.Device->supportsCounterSampling(MTL::CounterSamplingPointAtStageBoundary)) return nullptr;
    auto *counter_set = TimestampCounterSet(ctx);
    if (!counter_set) return nullptr;

    const auto descriptor = NS::TransferPtr(MTL::CounterSampleBufferDescriptor::alloc()->init());
    descriptor->setCounterSet(counter_set);
    descriptor->setSampleCount(max_passes * 2);
    descriptor->setStorageMode(MTL::StorageModeShared);
    NS::Error *error = nullptr;
    auto buffer = NS::TransferPtr(ctx.Device->newCounterSampleBuffer(descriptor.get(), &error));
    if (!buffer) return nullptr;
    return std::unique_ptr<PassTimer>{new PassTimer{std::move(buffer), max_passes}};
}

std::optional<uint32_t> PassTimer::Claim(std::string_view name) {
    if (Names.size() >= MaxPasses) return {};
    Names.emplace_back(name);
    return uint32_t(Names.size() - 1);
}

std::vector<PassTimer::Pass> PassTimer::Resolve() {
    std::vector<Pass> passes;
    if (Names.empty()) return passes;

    auto *data = SampleBuffer->resolveCounterRange(NS::Range::Make(0, Names.size() * 2));
    if (!data) return passes;
    const auto *timestamps = static_cast<const MTL::CounterResultTimestamp *>(data->bytes());
    if (!timestamps || data->length() < Names.size() * 2 * sizeof(MTL::CounterResultTimestamp)) return passes;

    passes.reserve(Names.size());
    for (size_t i = 0; i < Names.size(); ++i) {
        const auto start = timestamps[i * 2].timestamp, end = timestamps[i * 2 + 1].timestamp;
        // Unwritten and wrapped sample pairs are unusable.
        if (start == MTL::CounterErrorValue || end == MTL::CounterErrorValue || end < start) continue;
        passes.emplace_back(Names[i], float(end - start) * 1e-6f);
    }
    return passes;
}
} // namespace mtl
