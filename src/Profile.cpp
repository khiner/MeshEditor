#include "Profile.h"

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
    uint32_t Stat;
    std::chrono::steady_clock::time_point Start;
};
// Key samples by tree position to distinguish identical scopes reached through different call chains.
struct Stat {
    std::string_view Name;
    uint32_t Parent;
    uint32_t Depth;
    std::string Path;
    std::vector<float> Ms;
};
constexpr uint32_t NoParent{~0u};
struct Summary {
    float Total{}, Mean{}, Median{}, P95{}, P99{}, Min{}, Max{};
};

bool Recording{false};
std::unique_ptr<mtl::PassTimer> Timer;
std::vector<CpuSpan> OpenCpu;
std::vector<Stat> GpuStats, CpuStats, Counters;

float Total(const std::vector<float> &ms) { return std::accumulate(ms.begin(), ms.end(), 0.f); }

Summary Summarize(std::vector<float> sorted) {
    if (sorted.empty()) return {};
    std::ranges::sort(sorted);
    const auto percentile = [&](float p) { return sorted[std::min(sorted.size() - 1, size_t(std::ceil(p * float(sorted.size() - 1))))]; };
    const auto total = Total(sorted);
    return {total, total / float(sorted.size()), percentile(0.5f), percentile(0.95f), percentile(0.99f), sorted.front(), sorted.back()};
}

// Return a scope index, preserving first-opened report order.
uint32_t StatIndex(std::vector<Stat> &stats, std::string_view name, uint32_t parent) {
    const auto it = std::ranges::find_if(stats, [&](const Stat &stat) { return stat.Parent == parent && stat.Name == name; });
    if (it != stats.end()) return uint32_t(it - stats.begin());
    const auto depth = parent == NoParent ? 0u : stats[parent].Depth + 1u;
    auto path = parent == NoParent ? std::string{name} : stats[parent].Path + '/' + std::string{name};
    stats.emplace_back(name, parent, depth, std::move(path), std::vector<float>{});
    return uint32_t(stats.size() - 1);
}

void PrintSubtree(const std::vector<Stat> &stats, const std::vector<std::vector<uint32_t>> &children, uint32_t i, bool print_gaps, auto &&row, auto &&gap_row) {
    const auto summary = Summarize(stats[i].Ms);
    row(stats[i].Depth, stats[i].Name, stats[i].Ms.size(), summary);
    float claimed = 0;
    for (const auto child : children[i]) {
        PrintSubtree(stats, children, child, print_gaps, row, gap_row);
        claimed += Total(stats[child].Ms);
    }
    if (print_gaps && !children[i].empty()) gap_row(stats[i].Depth + 1u, Total(stats[i].Ms) - claimed);
}

// Print unattributed parent time when child scopes are nested inside the parent.
void ReportTable(std::string_view title, const std::vector<Stat> &stats, bool children_run_inside_parent) {
    if (stats.empty()) return;
    std::vector<uint32_t> roots;
    std::vector<std::vector<uint32_t>> children(stats.size());
    for (uint32_t i = 0; i < stats.size(); ++i) {
        if (stats[i].Parent == NoParent) roots.emplace_back(i);
        else children[stats[i].Parent].emplace_back(i);
    }
    float wall = 0;
    for (const auto root : roots) wall += Total(stats[root].Ms);
    const auto share = [&](float ms) { return wall > 0 ? 100.f * ms / wall : 0.f; };

    std::println("\n{}", title);
    std::println("  {:<28} {:>6} {:>9} {:>9} {:>9} {:>9} {:>10} {:>7}", "scope", "calls", "mean", "median", "min", "max", "total", "%");
    const auto row = [&](uint32_t depth, std::string_view name, size_t calls, const Summary &summary) {
        std::println(
            "  {:<28} {:>6} {:>9.3f} {:>9.3f} {:>9.3f} {:>9.3f} {:>10.1f} {:>7.1f}",
            std::string(depth * 2, ' ') + std::string{name}, calls, summary.Mean, summary.Median, summary.Min, summary.Max, summary.Total, share(summary.Total)
        );
    };
    const auto gap_row = [&](uint32_t depth, float ms) {
        std::println(
            "  {:<28} {:>6} {:>9} {:>9} {:>9} {:>9} {:>10.1f} {:>7.1f}",
            std::string(depth * 2, ' ') + "unattributed", "-", "-", "-", "-", "-", ms, share(ms)
        );
    };
    for (const auto root : roots) PrintSubtree(stats, children, root, children_run_inside_parent, row, gap_row);
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
            "    {{\"name\":\"{}\",\"path\":\"{}\",\"depth\":{},\"samples\":{},\"mean\":{:.6f},\"median\":{:.6f},\"p95\":{:.6f},\"p99\":{:.6f},\"min\":{:.6f},\"max\":{:.6f},\"total\":{:.6f}}}{}\n",
            stats[i].Name, stats[i].Path, stats[i].Depth, stats[i].Ms.size(), summary.Mean, summary.Median, summary.P95, summary.P99, summary.Min, summary.Max, summary.Total,
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
    Recording = false;
    Timer.reset();
}

void BeginRecording() {
    if (!Enabled) return;
    // Discard timer state from an interrupted recording.
    if (Timer) Timer->Reset();
    Recording = true;
}

mtl::PassTimer *RecordingTimer() { return Enabled ? Timer.get() : nullptr; }

void EndRecording() {
    if (!Enabled) return;
    Recording = false;
}

void BeginCpu(std::string_view name) {
    if (!Enabled) return;
    // Allocate the report slot before nested scopes close.
    OpenCpu.emplace_back(StatIndex(CpuStats, name, OpenCpu.empty() ? NoParent : OpenCpu.back().Stat), std::chrono::steady_clock::now());
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
    Counters[StatIndex(Counters, name, NoParent)].Ms.emplace_back(float(value));
}

void Resolve(MTL::CommandBuffer *command_buffer) {
    if (!Enabled || !command_buffer) return;
    const auto ms = float((command_buffer->GPUEndTime() - command_buffer->GPUStartTime()) * 1e3);
    const auto submit = StatIndex(GpuStats, "Submit", NoParent);
    if (ms > 0.f) GpuStats[submit].Ms.emplace_back(ms);
    if (!Timer) return;
    for (const auto &pass : Timer->Resolve()) GpuStats[StatIndex(GpuStats, pass.Name, submit)].Ms.emplace_back(pass.Ms);
}

void ClearStats() {
    assert(OpenCpu.empty() && !Recording && "Profile: clear with a scope open or a recording active");
    GpuStats.clear();
    CpuStats.clear();
    Counters.clear();
}

void ReportCpuPhase(std::string_view title) {
    if (!Enabled) return;
    ReportTable(title, CpuStats, true);
}

void Report() {
    // GPU pass intervals can overlap siblings and submission parents, so the report does not partition them like nested CPU scopes.
    ReportTable("GPU pass timings (hardware timestamps)", GpuStats, false);
    ReportTable("CPU timings", CpuStats, true);
    ReportCounterTable(Counters);
    ReportJson();
}
} // namespace profile
