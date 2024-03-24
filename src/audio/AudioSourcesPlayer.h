#pragma once

#include <vector>

struct AudioSource;

struct AudioSourcesPlayer {
    AudioSourcesPlayer();
    ~AudioSourcesPlayer();

    void Add(std::unique_ptr<AudioSource> source);

    void Start();
    void Stop();

private:
    std::vector<std::unique_ptr<AudioSource>> AudioSources;
};
