#pragma once

#include <optional>
#include <string>

// The output device and the rate requested of it, a user preference kept in the user data directory.
struct AudioOutputConfig {
    std::string DeviceName; // Empty selects the system default.
    uint32_t SampleRate{0}; // Zero selects the device default.
    bool operator==(const AudioOutputConfig &) const = default;
};

std::optional<AudioOutputConfig> LoadAudioOutputConfig();
bool SaveAudioOutputConfig(const AudioOutputConfig &);
