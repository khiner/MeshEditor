#include "action/Audio.h"
#include "audio/AudioDevice.h"
#include "audio/AudioOutputConfig.h"
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

    if (BeginCombo("Output device", config.DeviceName.empty() ? "System default" : config.DeviceName.c_str())) {
        for (const auto &name : info.OutDeviceNames) {
            const bool is_selected = name == config.DeviceName;
            if (Selectable(name.c_str(), is_selected) && !is_selected) action::Emit(action::audio::SetOutputDevice{name});
            if (is_selected) SetItemDefaultFocus();
        }
        EndCombo();
    }
    if (BeginCombo("Sample rate", SampleRateName(info, info.SampleRate).c_str())) {
        for (const uint32_t option : info.NativeSampleRates) {
            const bool is_selected = option == info.SampleRate;
            if (Selectable(SampleRateName(info, option).c_str(), is_selected) && !is_selected) action::Emit(action::audio::SetOutputDevice{config.DeviceName, option});
            if (is_selected) SetItemDefaultFocus();
        }
        EndCombo();
    }

    f.Draw<&AudioOutputMix::Muted>();
    SameLine();
    if (mix.Muted) BeginDisabled();
    f.Draw<&AudioOutputMix::Volume>();
    if (mix.Muted) EndDisabled();
}
