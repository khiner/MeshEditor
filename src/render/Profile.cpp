#include "render/Profile.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <numeric>
#include <print>
#include <vector>

namespace profile {
namespace {
// Enough for every pass in one recording, twice over.
constexpr uint32_t MaxQueries{128};

struct GpuSpan {
    std::string_view Name;
    uint32_t First, Last, Depth;
};
struct CpuSpan {
    uint32_t Stat; // Index into CpuStats.
    std::chrono::steady_clock::time_point Start;
};
// Every sample one scope name collected across the run.
struct Stat {
    std::string_view Name;
    uint32_t Depth;
    std::vector<float> Ms;
};

vk::UniqueQueryPool Pool;
float Period{1}; // Nanoseconds per timestamp tick.
uint32_t Count{0}; // Queries written into the current recording.
vk::CommandBuffer RecordingCb{}; // Non-null between BeginRecording and EndRecording.
std::vector<GpuSpan> GpuSpans;
std::vector<uint32_t> OpenGpu; // Indices into GpuSpans, innermost last.
std::vector<CpuSpan> OpenCpu; // Innermost last.
// First-seen order, which is the order the work runs in.
std::vector<Stat> GpuStats, CpuStats;

float Total(const std::vector<float> &ms) { return std::accumulate(ms.begin(), ms.end(), 0.f); }

// Index of `name`'s samples, created on first mention so report order follows opening order.
uint32_t StatIndex(std::vector<Stat> &stats, std::string_view name, uint32_t depth) {
    const auto it = std::ranges::find(stats, name, &Stat::Name);
    if (it != stats.end()) return uint32_t(it - stats.begin());
    stats.emplace_back(name, depth, std::vector<float>{});
    return uint32_t(stats.size() - 1);
}

void ReportTable(std::string_view title, const std::vector<Stat> &stats) {
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

void ResetRecording() {
    Count = 0;
    GpuSpans.clear();
    OpenGpu.clear();
}
} // namespace

void Init(vk::Device device, vk::PhysicalDevice physical_device) {
    Pool = device.createQueryPoolUnique({{}, vk::QueryType::eTimestamp, MaxQueries});
    Period = physical_device.getProperties().limits.timestampPeriod;
}

void Deinit() { Pool.reset(); }

void BeginRecording(vk::CommandBuffer cb) {
    if (!Enabled) return;
    // A recording that never reached a submit leaves its spans behind.
    ResetRecording();
    RecordingCb = cb;
    cb.resetQueryPool(*Pool, 0, MaxQueries);
    BeginGpu("Submit");
}

void EndRecording() {
    EndGpu();
    RecordingCb = nullptr;
}

void BeginGpu(std::string_view name) {
    if (!Enabled || !RecordingCb) return;
    assert(Count + 2 <= MaxQueries && "Profile: raise MaxQueries");
    GpuSpans.emplace_back(name, Count, 0u, uint32_t(OpenGpu.size()));
    OpenGpu.emplace_back(uint32_t(GpuSpans.size() - 1));
    RecordingCb.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, *Pool, Count++);
}

void EndGpu() {
    if (!Enabled || !RecordingCb) return;
    assert(!OpenGpu.empty() && "Profile: GPU scope closed without opening");
    GpuSpans[OpenGpu.back()].Last = Count;
    OpenGpu.pop_back();
    RecordingCb.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, *Pool, Count++);
}

void BeginCpu(std::string_view name) {
    if (!Enabled) return;
    // Claim the report slot now: an inner scope closes first.
    OpenCpu.emplace_back(StatIndex(CpuStats, name, uint32_t(OpenCpu.size())), std::chrono::steady_clock::now());
}

void EndCpu() {
    if (!Enabled) return;
    assert(!OpenCpu.empty() && "Profile: CPU scope closed without opening");
    const auto span = OpenCpu.back();
    OpenCpu.pop_back();
    const std::chrono::duration<float, std::milli> elapsed = std::chrono::steady_clock::now() - span.Start;
    CpuStats[span.Stat].Ms.emplace_back(elapsed.count());
}

void Resolve(vk::Device device) {
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

void ClearStats() {
    assert(OpenCpu.empty() && !RecordingCb && "Profile: clear with a scope open or a recording active");
    GpuStats.clear();
    CpuStats.clear();
}

void Report() {
    ReportTable("GPU pass timings", GpuStats);
    ReportTable("CPU timings", CpuStats);
}
} // namespace profile
