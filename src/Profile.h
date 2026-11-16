#pragma once

#include "metal/PassTimer.h"

#include <filesystem>

// Profiling is compiled in when the build defines PROFILE_BUILD.
namespace profile {
#ifdef PROFILE_BUILD
inline bool Enabled{false};
inline std::filesystem::path JsonPath{};
inline void Enable(bool on) { Enabled = on; }

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
#else
inline constexpr bool Enabled{false};

inline void Enable(bool) {}
inline void Init(const mtl::Context &) {}
inline void Deinit() {}
inline void BeginRecording() {}
inline mtl::PassTimer *RecordingTimer() { return nullptr; }
inline void EndRecording() {}
inline void RecordCounter(std::string_view, double) {}
inline void Resolve(const mtl::Completion &) {}
inline void ClearStats() {}
inline void ReportCpuPhase(std::string_view) {}
inline void Report() {}

struct CpuScope {
    CpuScope(std::string_view) {}
};
#endif
} // namespace profile
