#pragma once

#include "CreateSvgResource.h"

#include <entt/entity/fwd.hpp>

#include <filesystem>

struct Scene;

struct AcousticScene {
    AcousticScene(entt::registry &);
    ~AcousticScene();

    static void LoadRealImpact(const std::filesystem::path &directory, entt::registry &, Scene &, CreateSvgResource);

    void RenderControls(Scene &);

    entt::registry &R;
};
