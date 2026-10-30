#pragma once

#include "metal/PassTimer.h"

#include <filesystem>

namespace profile {
inline bool Enabled{false};
inline std::filesystem::path JsonPath{};

void Init(const mtl::Context &);
void Deinit();

void BeginRecording();
mtl::PassTimer *RecordingTimer();
void EndRecording();

void BeginCpu(std::string_view name);
void EndCpu();
void RecordCounter(std::string_view name, double value);

// Resolve timing from a completed command's feedback.
void Resolve(const mtl::Completion &);
// Clear samples while no scope or recording is active.
void ClearStats();
void ReportCpuPhase(std::string_view title);
void Report();

// Time CPU work and GPU waits for this scope.
struct CpuScope {
    CpuScope(std::string_view name) { BeginCpu(name); }
    ~CpuScope() { EndCpu(); }
};
} // namespace profile
