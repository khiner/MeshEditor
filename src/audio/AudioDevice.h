#pragma once

#include "audio/AudioSchema.h"
#include "state/Entity.h"

#include <atomic>
#include <vector>

struct AudioOutputConfig;
struct AudioOutputMix;

struct AudioDeviceResource {
    AudioDeviceResource(state::Registry &);
    ~AudioDeviceResource();
    AudioDeviceResource(const AudioDeviceResource &) = delete;
    AudioDeviceResource &operator=(const AudioDeviceResource &) = delete;

    state::Registry *R;

    uint32_t SampleRate{0}; // Negotiated output rate.
    // The scheduling group the device's IO thread belongs to, for threads rendering alongside it.
    // Null when the host publishes none.
    void *RenderWorkgroup{nullptr};
    std::vector<std::string> OutDeviceNames;
    std::vector<uint32_t> NativeSampleRates;
    bool Initialized{false};
    bool Started{false};

    // Opaque CoreAudio handles keep platform headers out of every user of this resource.
    void *OutputUnit{nullptr};

    // Main/UI thread writes TargetGain; only CoreAudio's render thread touches RenderGain.
    // A block ramp avoids clicks without locks or work on the control thread.
    std::atomic<float> TargetGain{1.f};
    float RenderGain{1.f};

    // Device-open configuration used to detect changes requiring a reopen.
    std::string DeviceName; // Empty selects the system default.
    uint32_t RequestedSampleRate{0}; // 0 selects the device default.
};

void ReconcileAudioDevice(AudioDeviceResource &, const AudioOutputConfig &, const AudioOutputMix &);
void ApplyAudioMix(AudioDeviceResource &, const AudioOutputMix &);
