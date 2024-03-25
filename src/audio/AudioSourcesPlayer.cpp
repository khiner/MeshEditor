#include "AudioSourcesPlayer.h"

#include <format>
#include <string_view>

#include "AudioSource.h"

#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include "imgui.h"

enum IO_ {
    IO_None = -1,
    IO_In,
    IO_Out
};
using IO = IO_;
constexpr IO IO_All[] = {IO_In, IO_Out};
constexpr int IO_Count = 2;

void data_callback(ma_device *device, void *output, const void *input, ma_uint32 frame_count) {
    ma_waveform_read_pcm_frames((ma_waveform *)device->pUserData, output, frame_count, nullptr);

    (void)input; // Unused
}

const std::vector<uint> PrioritizedSampleRates = {std::begin(g_maStandardSampleRatePriorities), std::end(g_maStandardSampleRatePriorities)};

ma_context AudioContext;
ma_device Device;
ma_waveform Sinewave;

std::vector<ma_device_info *> DeviceInfos[IO_Count];
std::vector<std::string> DeviceNames[IO_Count];
std::vector<uint> NativeSampleRates[IO_Count];

const ma_device_info *GetDeviceInfo(IO io, std::string_view device_name) {
    for (const ma_device_info *info : DeviceInfos[io]) {
        if (info->name == device_name) return info;
    }
    return nullptr;
}
const ma_device_id *GetDeviceId(IO io, std::string_view device_name) {
    const auto *device_info = GetDeviceInfo(io, device_name);
    return device_info ? &device_info->id : nullptr;
}

std::string GetSampleRateName(IO io, const uint sample_rate) {
    const bool is_native = std::find(NativeSampleRates[io].begin(), NativeSampleRates[io].end(), sample_rate) != NativeSampleRates[io].end();
    return std::format("{}{}", sample_rate, is_native ? "*" : "");
}

AudioSourcesPlayer::AudioSourcesPlayer() {
    Stop();
    Init();
}

AudioSourcesPlayer::~AudioSourcesPlayer() {
    Uninit();
}

void AudioSourcesPlayer::Add(std::unique_ptr<AudioSource> source) {
    AudioSources.emplace_back(std::move(source));
}

void AudioSourcesPlayer::Init() {
    if (ma_context_init(nullptr, 0, nullptr, &AudioContext) != MA_SUCCESS) throw std::runtime_error(std::format("Failed to initialize audio context."));
    static uint PlaybackDeviceCount, CaptureDeviceCount;
    static ma_device_info *PlaybackDeviceInfos, *CaptureDeviceInfos;
    if (ma_context_get_devices(&AudioContext, &PlaybackDeviceInfos, &PlaybackDeviceCount, &CaptureDeviceInfos, &CaptureDeviceCount)) {
        throw std::runtime_error("Failed to get audio devices.");
    }
    for (auto io : IO_All) {
        DeviceInfos[io].clear();
        DeviceNames[io].clear();
        NativeSampleRates[io].clear();
    }
    for (uint i = 0; i < CaptureDeviceCount; ++i) {
        DeviceInfos[IO_In].emplace_back(&CaptureDeviceInfos[i]);
        DeviceNames[IO_In].push_back(CaptureDeviceInfos[i].name);
    }
    for (uint i = 0; i < PlaybackDeviceCount; ++i) {
        DeviceInfos[IO_Out].emplace_back(&PlaybackDeviceInfos[i]);
        DeviceNames[IO_Out].push_back(PlaybackDeviceInfos[i].name);
    }

    ma_device_config device_config = ma_device_config_init(ma_device_type_playback);
    device_config.playback.pDeviceID = GetDeviceId(IO_Out, OutDeviceName);
    device_config.playback.format = ma_format_f32;
    device_config.playback.channels = 2;
    device_config.sampleRate = SampleRate;
    device_config.dataCallback = data_callback;
    device_config.pUserData = &Sinewave;

    if (ma_device_init(NULL, &device_config, &Device) != MA_SUCCESS) {
        throw std::runtime_error("Failed to open audio output device.");
    }

    // `ma_context_get_devices` doesn't return native sample rates, so we need to get the device info for the specific device.
    ma_device_info out_device_info;
    if (ma_context_get_device_info(&AudioContext, ma_device_type_playback, device_config.playback.pDeviceID, &out_device_info) != MA_SUCCESS) {
        throw std::runtime_error("Failed to get audio output device info.");
    }
    for (uint i = 0; i < out_device_info.nativeDataFormatCount; ++i) {
        const auto &native_format = out_device_info.nativeDataFormats[i];
        NativeSampleRates[IO_Out].emplace_back(native_format.sampleRate);
    }

    OutDeviceName = out_device_info.name;
    SampleRate = Device.sampleRate;

    ma_waveform_config sinewave_config = ma_waveform_config_init(
        Device.playback.format, Device.playback.channels, Device.sampleRate, ma_waveform_type_sine, 0.2, 220
    );
    ma_waveform_init(&sinewave_config, &Sinewave);
}

void AudioSourcesPlayer::Start() {
    if (!ma_device_is_started(&Device) && ma_device_start(&Device) != MA_SUCCESS) {
        ma_device_uninit(&Device);
        throw std::runtime_error("Failed to start audio output device.");
    }
    On = true;
}

void AudioSourcesPlayer::Stop() {
    On = false;
    if (!ma_device_is_started(&Device)) return;

    if (ma_device_stop(&Device) != MA_SUCCESS) {
        ma_device_uninit(&Device);
        throw std::runtime_error("Failed to stop audio output device.");
    }
}

void AudioSourcesPlayer::Uninit() {
    ma_device_uninit(&Device);
    ma_waveform_uninit(&Sinewave);
}

void AudioSourcesPlayer::OnVolumeChange() { ma_device_set_master_volume(&Device, Muted ? 0 : Volume); }

void AudioSourcesPlayer::RestartDevice() {
    const bool was_on = On;
    Stop();
    Uninit();
    Init();
    if (was_on) Start();
}

using namespace ImGui;

void AudioSourcesPlayer::RenderControls() {
    if (Checkbox("On", &On)) {
        if (On) Start();
        else Stop();
    }
    if (!On) {
        TextUnformatted("Audio device: Not started");
        return;
    }

    if (BeginCombo("Output device", OutDeviceName.c_str())) {
        for (const auto &option : DeviceNames[IO_Out]) {
            const bool is_selected = option == OutDeviceName;
            if (Selectable(option.c_str(), is_selected) && !is_selected) {
                OutDeviceName = option;
                SampleRate = 0; // Use the default sample rate when changing devices.
                RestartDevice();
            }
            if (is_selected) SetItemDefaultFocus();
        }
        EndCombo();
    }
    if (BeginCombo("Sample rate", GetSampleRateName(IO_Out, SampleRate).c_str())) {
        for (uint option : NativeSampleRates[IO_Out]) {
            const bool is_selected = option == SampleRate;
            if (Selectable(GetSampleRateName(IO_Out, option).c_str(), is_selected) && !is_selected) {
                SampleRate = option;
                RestartDevice();
            }
            if (is_selected) SetItemDefaultFocus();
        }
        EndCombo();
    }

    if (Checkbox("Muted", &Muted)) OnVolumeChange();
    SameLine();
    if (Muted) BeginDisabled();
    if (SliderFloat("Volume", &Volume, 0, 1, nullptr)) OnVolumeChange();
    if (Muted) EndDisabled();
}
