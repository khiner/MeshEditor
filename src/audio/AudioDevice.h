#pragma once

#include <functional>
#include <string>

using uint = uint32_t;

struct AudioDevice {
    using audio_callback_t = std::function<void(uint sample_rate, uint channels, float *output, const float *input, uint frames)>;

    AudioDevice(audio_callback_t);
    ~AudioDevice();

    void Init();
    void Start();
    void Stop();
    void Uninit();

    void RenderControls();

private:
    audio_callback_t Callback;

    bool On{false}, Muted{false};
    float Volume{1.0};
    std::string OutDeviceName;
    uint SampleRate{48000};

    void OnVolumeChange();
    void RestartDevice();
};
