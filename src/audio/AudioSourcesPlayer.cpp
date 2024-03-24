#include "AudioSourcesPlayer.h"

#include "AudioSource.h"

#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

void data_callback(ma_device *device, void *output, const void *input, ma_uint32 frame_count) {
    ma_waveform_read_pcm_frames((ma_waveform *)device->pUserData, output, frame_count, nullptr);

    (void)input; // Unused
}

void AudioSourcesPlayer::Add(std::unique_ptr<AudioSource> source) {
    AudioSources.emplace_back(std::move(source));
}

ma_device device;
ma_waveform sinewave;

AudioSourcesPlayer::AudioSourcesPlayer() {
    ma_device_config device_config = ma_device_config_init(ma_device_type_playback);
    device_config.playback.format = ma_format_f32;
    device_config.playback.channels = 2;
    device_config.sampleRate = 48000;
    device_config.dataCallback = data_callback;
    device_config.pUserData = &sinewave;

    if (ma_device_init(NULL, &device_config, &device) != MA_SUCCESS) {
        throw std::runtime_error("Failed to open audio playback device.");
    }

    ma_waveform_config sinewave_config = ma_waveform_config_init(
        device.playback.format, device.playback.channels, device.sampleRate, ma_waveform_type_sine, 0.2, 220
    );
    ma_waveform_init(&sinewave_config, &sinewave);
}

AudioSourcesPlayer::~AudioSourcesPlayer() {
    ma_device_uninit(&device);
    ma_waveform_uninit(&sinewave);
}

void AudioSourcesPlayer::Start() {
    if (ma_device_start(&device) != MA_SUCCESS) {
        ma_device_uninit(&device);
        throw std::runtime_error("Failed to start audio playback device.");
    }
}

void AudioSourcesPlayer::Stop() {
    if (ma_device_stop(&device) != MA_SUCCESS) {
        ma_device_uninit(&device);
        throw std::runtime_error("Failed to stop audio playback device.");
    }
}
