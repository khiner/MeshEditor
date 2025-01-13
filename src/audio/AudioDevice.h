#pragma once

#include <string>

#include "AudioBuffer.h"

struct AudioDevice {
    using audio_callback_t = void (*)(AudioBuffer);

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
    void Restart();
};
