#pragma once

#include <functional>
#include <string>

#include "FrameInfo.h"

struct AudioDevice {
    using audio_callback_t = std::function<void(FrameInfo, float *output, const float *input, uint32_t frames)>;

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
    uint32_t SampleRate{48'000};

    void OnVolumeChange();
    void RestartDevice();
};
