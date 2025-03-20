#include "AudioDevice.h"

#include "imgui.h"
#include "miniaudio.h"

#include <format>
#include <string_view>
#include <vector>

using uint = uint32_t;

namespace {
void DataCallback(ma_device *device, void *output, const void *input, ma_uint32 frame_count) {
    auto cb = reinterpret_cast<AudioDeviceCallback *>(device->pUserData);
    cb->Callback(AudioBuffer{device->sampleRate, device->playback.channels, frame_count, (const float *)input, (float *)output}, cb->UserData);
}

enum IO {
    IO_None = -1,
    IO_In,
    IO_Out
};
constexpr IO IO_All[] = {IO_In, IO_Out};
constexpr uint IO_Count = 2;

// Copied from `miniaudio.c::g_maStandardSampleRatePriorities`.
const std::vector<uint> PrioritizedSampleRates{
    ma_standard_sample_rate_48000,
    ma_standard_sample_rate_44100,

    ma_standard_sample_rate_32000,
    ma_standard_sample_rate_24000,
    ma_standard_sample_rate_22050,

    ma_standard_sample_rate_88200,
    ma_standard_sample_rate_96000,
    ma_standard_sample_rate_176400,
    ma_standard_sample_rate_192000,

    ma_standard_sample_rate_16000,
    ma_standard_sample_rate_11025,
    ma_standard_sample_rate_8000,

    ma_standard_sample_rate_352800,
    ma_standard_sample_rate_384000,
};

ma_context AudioContext;
ma_device Device;

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
} // namespace

AudioDevice::AudioDevice(AudioDeviceCallback data_callback) : Callback(std::move(data_callback)) {
    Init();
}
AudioDevice::~AudioDevice() {
    Uninit();
}

void AudioDevice::Init() {
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

    ma_device_config config = ma_device_config_init(ma_device_type_playback);
    config.playback.pDeviceID = GetDeviceId(IO_Out, OutDeviceName);
    config.playback.format = ma_format_f32;
    config.playback.channels = 1;
    config.sampleRate = SampleRate;
    config.dataCallback = DataCallback;
    config.pUserData = &Callback;

    if (ma_device_init(NULL, &config, &Device) != MA_SUCCESS) {
        throw std::runtime_error("Failed to open audio output device.");
    }

    // `ma_context_get_devices` doesn't return native sample rates, so we need to get the device info for the specific device.
    ma_device_info out_device_info;
    if (ma_context_get_device_info(&AudioContext, ma_device_type_playback, config.playback.pDeviceID, &out_device_info) != MA_SUCCESS) {
        throw std::runtime_error("Failed to get audio output device info.");
    }
    for (uint i = 0; i < out_device_info.nativeDataFormatCount; ++i) {
        const auto &native_format = out_device_info.nativeDataFormats[i];
        NativeSampleRates[IO_Out].emplace_back(native_format.sampleRate);
    }

    OutDeviceName = out_device_info.name;
    SampleRate = Device.sampleRate;
}

void AudioDevice::Start() {
    if (!ma_device_is_started(&Device) && ma_device_start(&Device) != MA_SUCCESS) {
        ma_device_uninit(&Device);
        throw std::runtime_error("Failed to start audio output device.");
    }
    On = true;
}

void AudioDevice::Stop() {
    On = false;
    if (!ma_device_is_started(&Device)) return;

    if (ma_device_stop(&Device) != MA_SUCCESS) {
        ma_device_uninit(&Device);
    }
}

void AudioDevice::Uninit() {
    Stop();
    ma_device_uninit(&Device);
}

void AudioDevice::OnVolumeChange() { ma_device_set_master_volume(&Device, Muted ? 0 : Volume); }

void AudioDevice::Restart() {
    const bool was_on = On;
    Uninit();
    Init();
    if (was_on) Start();
}

using namespace ImGui;

void AudioDevice::RenderControls() {
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
                Restart();
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
                Restart();
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
