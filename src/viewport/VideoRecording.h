#pragma once

#include "VideoRecorder.h"
#include "audio/AudioTypes.h"

// Present on the viewport entity iff recording is active.
struct VideoRecording {
    std::unique_ptr<VideoRecorder> Recorder;
    std::pair<uvec2, mtl::Extent2D> Region; // Locked at StartRecording.
    std::vector<float> Drained{}; // Scratch the captured audio is drained through each frame.
    // Set for a recording to be played back, which takes the master mix at the monitor's level rather than in pascals.
    bool Monitor{false};
    MonitorLimiter Limiter{}; // Recording-specific envelope that does not modify the device envelope.
    // Set when no device produces audio.
    // Each captured frame then renders its share on this thread.
    uint32_t OfflineRate{0};
    double OfflineCarry{0}; // Fractional frames owed, so a non-integral rate over fps stays in step.
    int Fps{0};
};
