#pragma once

#include "metal/PassTimer.h"

#include <filesystem>
#include <string_view>

// Store CPU and GPU samples for the next report.
namespace profile {
inline bool Enabled{false};
inline std::filesystem::path JsonPath{};

void Init(const mtl::Context &);
void Deinit();

// Begin per-pass timing for a command buffer.
void BeginRecording();
mtl::PassTimer *RecordingTimer();
void EndRecording();

void BeginCpu(std::string_view name);
void EndCpu();
void RecordCounter(std::string_view name, double value);

// Resolve timing after the command buffer completes.
void Resolve(MTL::CommandBuffer *);
// Clear samples while no scope or recording is active.
void ClearStats();
// Report the current CPU scopes under title.
void ReportCpuPhase(std::string_view title);
void Report();

// Time CPU work and GPU waits for this scope.
struct CpuScope {
    CpuScope(std::string_view name) { BeginCpu(name); }
    ~CpuScope() { EndCpu(); }
};
} // namespace profile
