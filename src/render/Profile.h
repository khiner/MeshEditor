#pragma once

#include <chrono>
#include <string_view>
#include <vector>
#include <vulkan/vulkan.hpp>

// Where a frame's time goes, gathered over a run and reported as a summary.
// GPU times come from timestamp queries, CPU times from the wall clock.
struct Profile {
    // Resolving timestamps costs a readback per submit.
    static inline bool Enabled{false};

    Profile(vk::Device, vk::PhysicalDevice);

    // Bracket one recording, inside the command buffer's begin and end. Their span is the
    // submit's total, which every GPU scope inside divides up.
    void BeginRecording(vk::CommandBuffer);
    void EndRecording(vk::CommandBuffer);

    void BeginGpu(vk::CommandBuffer, std::string_view name);
    void EndGpu(vk::CommandBuffer);
    void BeginCpu(std::string_view name);
    void EndCpu();

    // Fold the finished submit's timestamps into the run. Call once its fence has signaled.
    void Resolve(vk::Device);
    void Report() const;

private:
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

    // Index of `name`'s samples, created on first mention so report order follows opening order.
    static uint32_t StatIndex(std::vector<Stat> &, std::string_view name, uint32_t depth);
    static void ReportTable(std::string_view title, const std::vector<Stat> &);

    void ResetRecording();

    vk::UniqueQueryPool Pool;
    float Period{1}; // Nanoseconds per timestamp tick.
    uint32_t Count{0}; // Queries written into the current recording.
    std::vector<GpuSpan> GpuSpans;
    std::vector<uint32_t> OpenGpu; // Indices into GpuSpans, innermost last.
    std::vector<CpuSpan> OpenCpu; // Innermost last.
    // First-seen order, which is the order the work runs in.
    std::vector<Stat> GpuStats, CpuStats;
};

// Times a pass on the GPU while in scope. Open it outside a render pass: timestamps land on
// command encoder boundaries, so one opened inside spans the whole pass.
struct GpuScope {
    GpuScope(Profile &p, vk::CommandBuffer cb, std::string_view name) : P{p}, Cb{cb} { P.BeginGpu(Cb, name); }
    ~GpuScope() { P.EndGpu(Cb); }

private:
    Profile &P;
    vk::CommandBuffer Cb;
};

// Times work on the CPU while in scope, including any GPU waits.
struct CpuScope {
    CpuScope(Profile &p, std::string_view name) : P{p} { P.BeginCpu(name); }
    ~CpuScope() { P.EndCpu(); }

private:
    Profile &P;
};
