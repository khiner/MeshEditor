#pragma once

#include "metal/PassTimer.h"

#include <filesystem>
#include <string_view>

// CPU and GPU timings gathered over a run and reported as a summary.
namespace profile {
inline bool Enabled{false};
inline std::filesystem::path JsonPath{};

void Init(const mtl::Context &);
void Deinit();

// Bracket a command buffer recording for per-pass timing.
void BeginRecording();
mtl::PassTimer *RecordingTimer();
void EndRecording();

void BeginCpu(std::string_view name);
void EndCpu();
void RecordCounter(std::string_view name, double value);

// Call after the command buffer completes.
void Resolve(MTL::CommandBuffer *);
// Drop every collected sample. Call with no scope open and no recording active.
void ClearStats();
// Print the CPU scopes collected so far under `title`, for a phase the frame report clears.
void ReportCpuPhase(std::string_view title);
void Report();

// Times work on the CPU while in scope, including any GPU waits.
struct CpuScope {
    CpuScope(std::string_view name) { BeginCpu(name); }
    ~CpuScope() { EndCpu(); }
};
} // namespace profile
