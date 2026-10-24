#include "audio/AudioOutputConfig.h"

#include "File.h"
#include "Paths.h"

#include <zpp_bits.h>

namespace {
std::filesystem::path ConfigPath() { return Paths::UserData() / "audio_output.state"; }
} // namespace

std::optional<AudioOutputConfig> LoadAudioOutputConfig() {
    const auto bytes = File::Read(ConfigPath());
    if (!bytes) return std::nullopt;
    AudioOutputConfig config;
    zpp::bits::in in{*bytes};
    if (zpp::bits::failure(in(config)) || in.position() != bytes->size()) return std::nullopt;
    return config;
}

bool SaveAudioOutputConfig(const AudioOutputConfig &config) {
    std::vector<std::byte> bytes;
    if (zpp::bits::failure(zpp::bits::out{bytes}(config))) return false;
    return bool(File::WriteAtomic(ConfigPath(), bytes));
}
