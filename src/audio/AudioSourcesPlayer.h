#pragma once

#include <vector>

struct AudioSource;

struct AudioSourcesPlayer {
    AudioSourcesPlayer();
    ~AudioSourcesPlayer();

    void Add(std::unique_ptr<AudioSource> source);

    void Init();
    void Start();
    void Stop();
    void Uninit();

    void RenderControls();

private:
    std::vector<std::unique_ptr<AudioSource>> AudioSources;
    bool On{false}, Muted{false};
    float Volume{1.0};
    std::string OutDeviceName;
    uint SampleRate{48000};

    void OnVolumeChange();
    void RestartDevice();
};
