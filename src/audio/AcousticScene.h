#pragma once

#include "AudioBuffer.h"
#include "CreateSvgResource.h"

#include <entt/entity/fwd.hpp>

#include <filesystem>

struct Scene;

struct AcousticScene {
    AcousticScene(entt::registry &, CreateSvgResource);
    ~AcousticScene();

    void LoadRealImpact(const std::filesystem::path &directory, Scene &) const;

    void RenderControls(Scene &);

    void ProduceAudio(AudioBuffer) const;

    entt::registry &R;
    CreateSvgResource CreateSvg;
};