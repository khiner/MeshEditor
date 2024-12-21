#pragma once

#include <entt/entity/registry.hpp>

using uint = uint32_t;

struct AudioSourcesPlayer {
    AudioSourcesPlayer(entt::registry &);
    ~AudioSourcesPlayer();

    void Init();
    void Start();
    void Stop();
    void Uninit();

    void RenderControls();

private:
    entt::registry &R;

    bool On{false}, Muted{false};
    float Volume{1.0};
    std::string OutDeviceName;
    uint SampleRate{48000};

    void OnVolumeChange();
    void RestartDevice();
};
