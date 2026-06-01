#pragma once

#include "entt_fwd.h"
#include "render/SvgResource.h"

struct ViewportIconsTransform {
    std::unique_ptr<SvgResource> Select, SelectBox, Move, Rotate, Scale, Universal;
};
struct ViewportIconsShading {
    std::unique_ptr<SvgResource> Wireframe, Solid, MaterialPreview, Rendered;
};
struct AnimationIcons {
    std::unique_ptr<SvgResource> Play, Pause, JumpStart, JumpEnd;
};

// Loaded once by LoadViewportIcons after the GPU context is ready.
struct ViewportIcons {
    ViewportIconsTransform Transform;
    ViewportIconsShading Shading;
    std::unique_ptr<SvgResource> Overlay;
    AnimationIcons Anim;
};

// Emplaces the ViewportIcons context singleton and uploads each SVG into its bitmap texture.
void LoadViewportIcons(entt::registry &);
