#pragma once

#include <string_view>
#include <vulkan/vulkan.hpp>

// Where a frame's time goes, gathered over a run and reported as a summary.
// GPU times come from timestamp queries, CPU times from the wall clock.
namespace profile {
// Resolving timestamps costs a readback per submit.
inline bool Enabled{false};

// Create the timestamp query pool.
void Init(vk::Device, vk::PhysicalDevice);
// Release the query pool. Call before destroying the device.
void Deinit();

// Bracket one recording, inside the command buffer's begin and end. Their span is the
// submit's total, which every GPU scope inside divides up.
void BeginRecording(vk::CommandBuffer);
void EndRecording();

void BeginGpu(std::string_view name);
void EndGpu();
void BeginCpu(std::string_view name);
void EndCpu();

// Fold the finished submit's timestamps into the run. Call once its fence has signaled.
void Resolve(vk::Device);
// Drop every collected sample. Call with no scope open and no recording active.
void ClearStats();
void Report();

// Times a pass on the GPU while in scope. Open it outside a render pass: timestamps land on
// command encoder boundaries, so one opened inside spans the whole pass.
struct GpuScope {
    GpuScope(std::string_view name) { BeginGpu(name); }
    ~GpuScope() { EndGpu(); }
};

// Times work on the CPU while in scope, including any GPU waits.
struct CpuScope {
    CpuScope(std::string_view name) { BeginCpu(name); }
    ~CpuScope() { EndCpu(); }
};
} // namespace profile
