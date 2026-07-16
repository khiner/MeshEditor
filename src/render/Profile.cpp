#include "render/Profile.h"

#include <algorithm>
#include <cassert>
#include <numeric>
#include <print>

namespace {
// Enough for every pass in one recording, twice over.
constexpr uint32_t MaxQueries{128};

float Total(const std::vector<float> &ms) { return std::accumulate(ms.begin(), ms.end(), 0.f); }
} // namespace

Profile::Profile(vk::Device device, vk::PhysicalDevice physical_device)
    : Pool{device.createQueryPoolUnique({{}, vk::QueryType::eTimestamp, MaxQueries})},
      Period{physical_device.getProperties().limits.timestampPeriod} {}

void Profile::ResetRecording() {
    Count = 0;
    GpuSpans.clear();
    OpenGpu.clear();
}

void Profile::BeginRecording(vk::CommandBuffer cb) {
    if (!Enabled) return;
    // A recording that never reached a submit leaves its spans behind.
    ResetRecording();
    cb.resetQueryPool(*Pool, 0, MaxQueries);
    BeginGpu(cb, "Submit");
}

void Profile::EndRecording(vk::CommandBuffer cb) { EndGpu(cb); }

void Profile::BeginGpu(vk::CommandBuffer cb, std::string_view name) {
    if (!Enabled) return;
    assert(Count + 2 <= MaxQueries && "Profile: raise MaxQueries");
    GpuSpans.emplace_back(name, Count, 0u, uint32_t(OpenGpu.size()));
    OpenGpu.emplace_back(uint32_t(GpuSpans.size() - 1));
    cb.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, *Pool, Count++);
}

void Profile::EndGpu(vk::CommandBuffer cb) {
    if (!Enabled) return;
    assert(!OpenGpu.empty() && "Profile: GPU scope closed without opening");
    GpuSpans[OpenGpu.back()].Last = Count;
    OpenGpu.pop_back();
    cb.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, *Pool, Count++);
}

void Profile::BeginCpu(std::string_view name) {
    if (!Enabled) return;
    // Claim the report slot now: an inner scope closes first.
    OpenCpu.emplace_back(StatIndex(CpuStats, name, uint32_t(OpenCpu.size())), std::chrono::steady_clock::now());
}

void Profile::EndCpu() {
    if (!Enabled) return;
    assert(!OpenCpu.empty() && "Profile: CPU scope closed without opening");
    const auto span = OpenCpu.back();
    OpenCpu.pop_back();
    const std::chrono::duration<float, std::milli> elapsed = std::chrono::steady_clock::now() - span.Start;
    CpuStats[span.Stat].Ms.emplace_back(elapsed.count());
}

uint32_t Profile::StatIndex(std::vector<Stat> &stats, std::string_view name, uint32_t depth) {
    const auto it = std::ranges::find(stats, name, &Stat::Name);
    if (it != stats.end()) return uint32_t(it - stats.begin());
    stats.emplace_back(name, depth, std::vector<float>{});
    return uint32_t(stats.size() - 1);
}

void Profile::Resolve(vk::Device device) {
    if (!Enabled || Count == 0) return;
    std::vector<uint64_t> ticks(Count);
    const auto result = device.getQueryPoolResults(
        *Pool, 0, Count, ticks.size() * sizeof(uint64_t), ticks.data(), sizeof(uint64_t),
        vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait
    );
    if (result == vk::Result::eSuccess) {
        for (const auto &span : GpuSpans) {
            const auto ns = double(ticks[span.Last] - ticks[span.First]) * double(Period);
            GpuStats[StatIndex(GpuStats, span.Name, span.Depth)].Ms.emplace_back(float(ns * 1e-6));
        }
    }
    // Spans stay registered: a resubmitted recording rewrites its queries and resolves here again.
}

void Profile::ClearStats() {
    assert(OpenCpu.empty() && OpenGpu.empty() && "Profile: clear with a scope open");
    GpuStats.clear();
    CpuStats.clear();
}

void Profile::ReportTable(std::string_view title, const std::vector<Stat> &stats) {
    if (stats.empty()) return;
    // The scope with the largest total spans everything timed, so it is what the rest divide up.
    const auto root = std::ranges::max_element(stats, {}, [](const Stat &s) { return Total(s.Ms); });
    const auto root_ms = root == stats.end() ? 0.f : Total(root->Ms);
    const auto share = [&](float ms) { return root_ms > 0 ? 100.f * ms / root_ms : 0.f; };

    std::println("\n{}", title);
    std::println("  {:<28} {:>6} {:>9} {:>9} {:>9} {:>9} {:>10} {:>7}", "scope", "calls", "mean", "median", "min", "max", "total", "%");
    float claimed = 0;
    for (const auto &stat : stats) {
        auto sorted = stat.Ms;
        std::ranges::sort(sorted);
        const auto sum = Total(sorted);
        if (stat.Depth == root->Depth + 1) claimed += sum;
        const auto name = std::string(stat.Depth * 2, ' ') + std::string{stat.Name};
        std::println(
            "  {:<28} {:>6} {:>9.3f} {:>9.3f} {:>9.3f} {:>9.3f} {:>10.1f} {:>7.1f}",
            name, sorted.size(), sum / float(sorted.size()), sorted[sorted.size() / 2],
            sorted.front(), sorted.back(), sum, share(sum)
        );
    }
    // Encoder setup, and any gaps the scopes one level in did not claim.
    std::println(
        "  {:<28} {:>6} {:>9} {:>9} {:>9} {:>9} {:>10.1f} {:>7.1f}", "  unattributed", "-", "-", "-", "-", "-",
        root_ms - claimed, share(root_ms - claimed)
    );
}

void Profile::Report() const {
    ReportTable("GPU pass timings", GpuStats);
    ReportTable("CPU timings", CpuStats);
}
