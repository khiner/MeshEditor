#pragma once

#include <entt/entity/fwd.hpp>

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

struct AudioOutputConfig;
struct AudioOutputMix;

struct AudioDeviceResource {
    AudioDeviceResource(entt::registry &, entt::entity viewport);
    ~AudioDeviceResource();
    AudioDeviceResource(const AudioDeviceResource &) = delete;
    AudioDeviceResource &operator=(const AudioDeviceResource &) = delete;

    entt::registry *R;
    entt::entity Viewport;

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

    // Selection the live device was opened with, to reconcile against and reopen only on a real change.
    std::string DeviceName; // Empty selects the system default.
    uint32_t RequestedSampleRate{0}; // 0 selects the device default.
};

void ReconcileAudioDevice(AudioDeviceResource &, const AudioOutputConfig &, const AudioOutputMix &);
void ApplyAudioMix(AudioDeviceResource &, const AudioOutputMix &);
void DrawAudioDeviceControls(entt::registry &, entt::entity viewport);
