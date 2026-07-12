#pragma once

#include <entt/entity/fwd.hpp>

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
    std::vector<std::string> OutDeviceNames;
    std::vector<uint32_t> NativeSampleRates;
    bool Initialized{false};

    // Selection the live device was opened with, to reconcile against and reopen only on a real change.
    std::string DeviceName; // Empty selects the system default.
    uint32_t RequestedSampleRate{0}; // 0 selects the device default.
};

void ReconcileAudioDevice(AudioDeviceResource &, const AudioOutputConfig &, const AudioOutputMix &);
void ApplyAudioMix(AudioDeviceResource &, const AudioOutputMix &);
void DrawAudioDeviceControls(entt::registry &, entt::entity viewport);
