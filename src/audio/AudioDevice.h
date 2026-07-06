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
};

void ConfigureAudioDevice(AudioDeviceResource &, const AudioOutputConfig &, const AudioOutputMix &);
void ApplyAudioMix(AudioDeviceResource &, const AudioOutputMix &);
void DrawAudioDeviceControls(entt::registry &, entt::entity viewport);
