#pragma once

#include "project/Project.h"
#include "state/Registry.h"

// A scene with its project and engine, plus the audio system when requested, torn down in reverse.
struct Engine {
    state::Registry R;
    std::unique_ptr<project::Project> P;
    state::Entity Viewport{state::Null};
    bool Audio;

    explicit Engine(bool audio);
    ~Engine();
};
