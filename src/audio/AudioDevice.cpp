#include "AudioDevice.h"

#include "AudioSystem.h"
#include "action/Audio.h" // Replace<AudioOutputConfig>
#include "action/Emit.h"
#include "ui/FieldEdit.h"

#include "imgui.h"
#include "miniaudio.h"

#include <entt/entity/registry.hpp>

#include <format>

template<> struct FieldLimits<&AudioOutputMix::Volume> : Within<0., 1.> {};

namespace {
ma_context Context;
ma_device Device;
bool ContextInitialized = false;

void DataCallback(ma_device *device, void *output, const void *input, ma_uint32 frame_count) {
    auto &res = *reinterpret_cast<AudioDeviceResource *>(device->pUserData);
    ProcessAudio(*res.R, res.Viewport, AudioBuffer{device->sampleRate, device->playback.channels, frame_count, static_cast<const float *>(input), static_cast<float *>(output)});
}

std::string SampleRateName(const AudioDeviceResource &res, uint32_t sample_rate) {
    const auto &rates = res.NativeSampleRates;
    const bool is_native = std::find(rates.begin(), rates.end(), sample_rate) != rates.end();
    return std::format("{}{}", sample_rate, is_native ? "*" : "");
}
} // namespace

AudioDeviceResource::AudioDeviceResource(entt::registry &r, entt::entity viewport) : R(&r), Viewport(viewport) {}
AudioDeviceResource::~AudioDeviceResource() {
    if (Initialized) ma_device_uninit(&Device);
    if (ContextInitialized) {
        ma_context_uninit(&Context);
        ContextInitialized = false;
    }
}

void ConfigureAudioDevice(AudioDeviceResource &res, const AudioOutputConfig &config, const AudioOutputMix &mix) {
    if (!ContextInitialized) {
        if (ma_context_init(nullptr, 0, nullptr, &Context) != MA_SUCCESS) throw std::runtime_error("Failed to initialize audio context.");
        ContextInitialized = true;
    }
    if (res.Initialized) {
        ma_device_uninit(&Device);
        res.Initialized = false;
    }

    ma_device_info *playback_infos, *capture_infos;
    ma_uint32 playback_count, capture_count;
    if (ma_context_get_devices(&Context, &playback_infos, &playback_count, &capture_infos, &capture_count) != MA_SUCCESS) throw std::runtime_error("Failed to get audio devices.");
    res.OutDeviceNames.clear();
    const ma_device_id *device_id = nullptr;
    for (ma_uint32 i = 0; i < playback_count; ++i) {
        res.OutDeviceNames.emplace_back(playback_infos[i].name);
        if (config.DeviceName == playback_infos[i].name) device_id = &playback_infos[i].id;
    }

    ma_device_config device_config = ma_device_config_init(ma_device_type_playback);
    device_config.playback.pDeviceID = device_id;
    device_config.playback.format = ma_format_f32;
    device_config.playback.channels = 1;
    device_config.sampleRate = config.SampleRate; // 0 = device default
    device_config.coreaudio.allowNominalSampleRateChange = MA_TRUE; // Drive the OS device rate instead of resampling.
    device_config.dataCallback = DataCallback;
    device_config.pUserData = &res;
    if (ma_device_init(nullptr, &device_config, &Device) != MA_SUCCESS) throw std::runtime_error("Failed to open audio output device.");
    res.Initialized = true;

    // `ma_context_get_devices` omits native rates, so query the picked device for its format list.
    res.NativeSampleRates.clear();
    if (ma_device_info info; ma_context_get_device_info(&Context, ma_device_type_playback, device_id, &info) == MA_SUCCESS) {
        for (ma_uint32 i = 0; i < info.nativeDataFormatCount; ++i) res.NativeSampleRates.emplace_back(info.nativeDataFormats[i].sampleRate);
    }
    res.SampleRate = Device.sampleRate;

    ApplyAudioMix(res, mix);
}

void ApplyAudioMix(AudioDeviceResource &res, const AudioOutputMix &mix) {
    if (!res.Initialized) return;
    ma_device_set_master_volume(&Device, mix.Muted ? 0.f : mix.Volume);
    if (const bool started = ma_device_is_started(&Device); mix.On && !started) {
        if (ma_device_start(&Device) != MA_SUCCESS) throw std::runtime_error("Failed to start audio output device.");
    } else if (!mix.On && started) {
        ma_device_stop(&Device);
    }
}

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
