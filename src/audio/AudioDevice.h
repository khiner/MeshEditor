#pragma once

#include "AudioBuffer.h"

#include <string>

struct AudioDeviceCallback {
    using callback_t = void (*)(AudioBuffer, void *user_data);

    callback_t Callback;
    void *UserData;
};

struct AudioDevice {
    AudioDevice(AudioDeviceCallback);
    ~AudioDevice();

    void Init();
    void Start();
    void Stop();
    void Uninit();

    void RenderControls();

private:
    AudioDeviceCallback Callback;

    bool On{false}, Muted{false};
    float Volume{1.0};
    std::string OutDeviceName;
    uint32_t SampleRate{48'000};

    void OnVolumeChange();
    void Restart();
};
