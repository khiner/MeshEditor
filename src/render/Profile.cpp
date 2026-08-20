#include "render/Profile.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <format>
#include <fstream>
#include <memory>
#include <numeric>
#include <print>
#include <string>
#include <vector>

namespace profile {
namespace {
struct CpuSpan {
    uint32_t Stat; // Index into CpuStats.
    std::chrono::steady_clock::time_point Start;
};
struct Stat {
    std::string_view Name;
    uint32_t Depth;
    std::vector<float> Ms;
};
struct Summary {
    float Total{}, Mean{}, Median{}, P95{}, P99{}, Min{}, Max{};
};

MTL::CommandBuffer *RecordingCb{nullptr}; // Non-null between BeginRecording and EndRecording.
std::unique_ptr<mtl::PassTimer> Timer; // Null when the device cannot sample counters.
std::vector<CpuSpan> OpenCpu; // Innermost last.
// First-seen order, which is the order the work runs in.
std::vector<Stat> GpuStats, CpuStats, Counters;

float Total(const std::vector<float> &ms) { return std::accumulate(ms.begin(), ms.end(), 0.f); }

Summary Summarize(std::vector<float> sorted) {
    if (sorted.empty()) return {};
    std::ranges::sort(sorted);
    const auto percentile = [&](float p) { return sorted[std::min(sorted.size() - 1, size_t(std::ceil(p * float(sorted.size() - 1))))]; };
    const auto total = Total(sorted);
    return {total, total / float(sorted.size()), percentile(0.5f), percentile(0.95f), percentile(0.99f), sorted.front(), sorted.back()};
}

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
        const auto summary = Summarize(stat.Ms);
        if (stat.Depth == root->Depth + 1) claimed += summary.Total;
        const auto name = std::string(stat.Depth * 2, ' ') + std::string{stat.Name};
        std::println(
            "  {:<28} {:>6} {:>9.3f} {:>9.3f} {:>9.3f} {:>9.3f} {:>10.1f} {:>7.1f}",
            name, stat.Ms.size(), summary.Mean, summary.Median, summary.Min, summary.Max, summary.Total, share(summary.Total)
        );
    }
    // Encoder setup, and any gaps the scopes one level in did not claim.
    std::println(
        "  {:<28} {:>6} {:>9} {:>9} {:>9} {:>9} {:>10.1f} {:>7.1f}", "  unattributed", "-", "-", "-", "-", "-",
        root_ms - claimed, share(root_ms - claimed)
    );
}

void ReportCounterTable(const std::vector<Stat> &stats) {
    if (stats.empty()) return;
    std::println("\nRenderer workload counters");
    std::println("  {:<32} {:>7} {:>14} {:>14} {:>14} {:>14}", "counter", "samples", "mean", "median", "min", "max");
    for (const auto &stat : stats) {
        const auto summary = Summarize(stat.Ms);
        std::println(
            "  {:<32} {:>7} {:>14.1f} {:>14.1f} {:>14.1f} {:>14.1f}",
            stat.Name, stat.Ms.size(), summary.Mean, summary.Median, summary.Min, summary.Max
        );
    }
}

void ReportJsonSection(std::ofstream &out, std::string_view name, const std::vector<Stat> &stats) {
    out << std::format("  \"{}\": [\n", name);
    for (size_t i = 0; i < stats.size(); ++i) {
        const auto summary = Summarize(stats[i].Ms);
        out << std::format(
            "    {{\"name\":\"{}\",\"depth\":{},\"samples\":{},\"mean\":{:.6f},\"median\":{:.6f},\"p95\":{:.6f},\"p99\":{:.6f},\"min\":{:.6f},\"max\":{:.6f},\"total\":{:.6f}}}{}\n",
            stats[i].Name, stats[i].Depth, stats[i].Ms.size(), summary.Mean, summary.Median, summary.P95, summary.P99, summary.Min, summary.Max, summary.Total,
            i + 1 == stats.size() ? "" : ","
        );
    }
    out << "  ]";
}

void ReportJson() {
    if (JsonPath.empty()) return;
    std::ofstream out{JsonPath};
    if (!out) {
        std::println(stderr, "Profile: could not write JSON to '{}'.", JsonPath.string());
        return;
    }
    out << "{\n";
    ReportJsonSection(out, "gpuMs", GpuStats);
    out << ",\n";
    ReportJsonSection(out, "cpuMs", CpuStats);
    out << ",\n";
    ReportJsonSection(out, "counters", Counters);
    out << "\n}\n";
}

} // namespace

void Init(const mtl::Context &ctx) {
    if (!Enabled) return;
    Timer = mtl::PassTimer::Create(ctx);
    if (!Timer) std::println(stderr, "Profile: this device cannot sample GPU counters, so passes go untimed.");
}

void Deinit() {
    RecordingCb = nullptr;
    Timer.reset();
}

void BeginRecording(MTL::CommandBuffer *command_buffer) {
    if (!Enabled) return;
    // A recording that never reached a submit leaves its claims behind.
    if (Timer) Timer->Reset();
    RecordingCb = command_buffer;
}

mtl::PassTimer *RecordingTimer() { return Enabled ? Timer.get() : nullptr; }

void EndRecording() {
    if (!Enabled) return;
    RecordingCb = nullptr;
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

void RecordCounter(std::string_view name, double value) {
    if (!Enabled) return;
    Counters[StatIndex(Counters, name, 0)].Ms.emplace_back(float(value));
}

void Resolve(MTL::CommandBuffer *command_buffer) {
    if (!Enabled || !command_buffer) return;
    const auto ms = float((command_buffer->GPUEndTime() - command_buffer->GPUStartTime()) * 1e3);
    if (ms > 0.f) GpuStats[StatIndex(GpuStats, "Submit", 0)].Ms.emplace_back(ms);
    if (!Timer) return;
    for (const auto &pass : Timer->Resolve()) GpuStats[StatIndex(GpuStats, pass.Name, 1)].Ms.emplace_back(pass.Ms);
}

void ClearStats() {
    assert(OpenCpu.empty() && !RecordingCb && "Profile: clear with a scope open or a recording active");
    GpuStats.clear();
    CpuStats.clear();
    Counters.clear();
}

void Report() {
    ReportTable("GPU pass timings (hardware timestamps)", GpuStats);
    ReportTable("CPU timings", CpuStats);
    ReportCounterTable(Counters);
    ReportJson();
}
} // namespace profile
