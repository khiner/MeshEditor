#include "action/Audio.h"
#include "audio/AudioDevice.h"
#include "audio/AudioOutputConfig.h"
#include "ui/ChoiceCombo.h"
#include "ui/FieldEdit.h"

#include <format>

namespace {
std::string SampleRateName(const AudioDeviceInfo &info, uint32_t sample_rate) {
    const auto &rates = info.NativeSampleRates;
    const bool is_native = std::ranges::contains(rates, sample_rate);
    return std::format("{}{}", sample_rate, is_native ? "*" : "");
}
} // namespace

void DrawAudioDeviceControls(state::Registry &r) {
    using namespace ImGui;
    const auto &config = r.Resources.get<const AudioOutputConfig>();
    const auto &mix = r.get<const AudioOutputMix>(state::Document);
    const auto &info = r.Resources.get<const AudioDeviceInfo>();
    ui::Edit f{r, state::Document};

    f.Draw<&AudioOutputMix::On>();
    if (!mix.On) {
        TextUnformatted("Audio device: Not started");
        return;
    }

    ui::ChoiceCombo("Output device", config.DeviceName, info.OutDeviceNames, [](const std::string &name) { return name.empty() ? "System default" : name.c_str(); }, [](const std::string &name) { action::Emit(action::audio::SetOutputDevice{name}); });
    ui::ChoiceCombo("Sample rate", info.SampleRate, info.NativeSampleRates, [&](uint32_t option) { return SampleRateName(info, option); }, [&](uint32_t option) { action::Emit(action::audio::SetOutputDevice{config.DeviceName, option}); });

    f.Draw<&AudioOutputMix::Muted>();
    SameLine();
    if (mix.Muted) BeginDisabled();
    f.Draw<&AudioOutputMix::Volume>();
    if (mix.Muted) EndDisabled();
}
