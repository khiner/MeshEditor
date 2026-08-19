#pragma once

#include "metal/PassTimer.h"

#include <string_view>

// Where a frame's time goes, gathered over a run and reported as a summary.
namespace profile {
inline bool Enabled{false};

void Init(const mtl::Context &);
void Deinit();

// Bracket a command buffer recording for per-pass timing.
void BeginRecording(MTL::CommandBuffer *);
mtl::PassTimer *RecordingTimer();
void EndRecording();

void BeginCpu(std::string_view name);
void EndCpu();

// Call after the command buffer completes.
void Resolve(MTL::CommandBuffer *);
// Drop every collected sample. Call with no scope open and no recording active.
void ClearStats();
void Report();

// Times work on the CPU while in scope, including any GPU waits.
struct CpuScope {
    CpuScope(std::string_view name) { BeginCpu(name); }
    ~CpuScope() { EndCpu(); }
};
} // namespace profile
