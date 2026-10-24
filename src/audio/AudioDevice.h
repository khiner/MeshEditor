#pragma once

#include "audio/AudioSchema.h"
#include "state/Entity.h"

#include <atomic>
#include <vector>

struct AudioOutputConfig;
struct AudioOutputMix;

// What the open output device reports, which the interface shows.
struct AudioDeviceInfo {
    uint32_t SampleRate{0}; // Negotiated output rate, zero without an open device.
    std::vector<std::string> OutDeviceNames;
    std::vector<uint32_t> NativeSampleRates;
};

struct AudioDeviceResource {
    AudioDeviceResource(state::Registry &);
    ~AudioDeviceResource();
    AudioDeviceResource(const AudioDeviceResource &) = delete;
    AudioDeviceResource &operator=(const AudioDeviceResource &) = delete;

    state::Registry *R;

    // The scheduling group the device's IO thread belongs to, for threads rendering alongside it.
    // Null when the host publishes none.
    void *RenderWorkgroup{nullptr};
    bool Started{false};

    // Opaque CoreAudio handles keep platform headers out of every user of this resource.
    uint32_t Device{}; // The open device's object id.
    void *OutputUnit{nullptr};
    std::atomic<uint32_t> ReportedSampleRate{0}; // The device's nominal rate, written by its property listener.

    // Main/UI thread writes TargetGain; only CoreAudio's render thread touches RenderGain.
    // A block ramp avoids clicks without locks or work on the control thread.
    std::atomic<float> TargetGain{1.f};
    float RenderGain{1.f};

    // Device-open configuration used to detect changes requiring a reopen.
    std::string DeviceName; // Empty selects the system default.
    uint32_t RequestedSampleRate{0}; // 0 selects the device default.
};

// Opens the configured device, requests the configured rate of it, and reopens the render unit at the rate the device reports.
// Returns whether the unit reopened.
bool ReconcileAudioDevice(AudioDeviceResource &, AudioDeviceInfo &, const AudioOutputConfig &);
void ApplyAudioMix(AudioDeviceResource &, const AudioOutputMix &);
