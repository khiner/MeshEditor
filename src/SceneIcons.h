#pragma once

#include "AnimationTimeline.h" // AnimationIcons
#include "SvgResource.h"

#include <entt/entity/fwd.hpp>

#include <memory>

struct SceneIconsTransform {
    std::unique_ptr<SvgResource> Select, SelectBox, Move, Rotate, Scale, Universal;
};
struct SceneIconsShading {
    std::unique_ptr<SvgResource> Wireframe, Solid, MaterialPreview, Rendered;
};

// Loaded once by LoadSceneIcons after the GPU context is ready.
struct SceneIcons {
    SceneIconsTransform Transform;
    SceneIconsShading Shading;
    std::unique_ptr<SvgResource> Overlay;
    AnimationIcons Anim;
};

// Emplaces SceneIcons on `scene_entity` and uploads each SVG into its bitmap texture.
void LoadSceneIcons(entt::registry &, entt::entity scene_entity);
