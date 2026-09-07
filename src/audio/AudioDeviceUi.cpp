#include "action/Audio.h"
#include "action/Emit.h"
#include "audio/AudioDevice.h"
#include "audio/AudioSystem.h"
#include "audio/AudioUi.h"
#include "ui/FieldEdit.h"
#include <format>

template<> struct FieldLimits<&AudioOutputMix::Volume> : Within<0., 1.> {};

namespace {
std::string SampleRateName(const AudioDeviceResource &res, uint32_t sample_rate) {
    const auto &rates = res.NativeSampleRates;
    const bool is_native = std::find(rates.begin(), rates.end(), sample_rate) != rates.end();
    return std::format("{}{}", sample_rate, is_native ? "*" : "");
}
} // namespace

void DrawAudioDeviceControls(entt::registry &r, entt::entity viewport) {
    using namespace ImGui;
    const auto &config = r.get<const AudioOutputConfig>(viewport);
    const auto &mix = r.get<const AudioOutputMix>(viewport);
    const auto &res = r.ctx().get<const AudioDeviceResource>();
    ui::Edit f{r, viewport};

    f.Check<&AudioOutputMix::On>("On");
    if (!mix.On) {
        TextUnformatted("Audio device: Not started");
        return;
    }

    if (BeginCombo("Output device", config.DeviceName.empty() ? "System default" : config.DeviceName.c_str())) {
        for (const auto &name : res.OutDeviceNames) {
            const bool is_selected = name == config.DeviceName;
            if (Selectable(name.c_str(), is_selected) && !is_selected) action::Emit(action::Replace<AudioOutputConfig>{.Entity = viewport, .Value = {.DeviceName = name, .SampleRate = 0}});
            if (is_selected) SetItemDefaultFocus();
        }
        EndCombo();
    }
    if (BeginCombo("Sample rate", SampleRateName(res, res.SampleRate).c_str())) {
        for (const uint32_t option : res.NativeSampleRates) {
            const bool is_selected = option == res.SampleRate;
            if (Selectable(SampleRateName(res, option).c_str(), is_selected) && !is_selected) f.Set<&AudioOutputConfig::SampleRate>(option);
            if (is_selected) SetItemDefaultFocus();
        }
        EndCombo();
    }

    f.Check<&AudioOutputMix::Muted>("Muted");
    SameLine();
    if (mix.Muted) BeginDisabled();
    f.Slider<&AudioOutputMix::Volume>("Volume");
    if (mix.Muted) EndDisabled();
}
