#pragma once

#include "ViewCamera.h"
#include "World.h"

#include "gpu/StudioLight.h"
#include "gpu/ViewportTheme.h"

#include <array>

struct SceneDefaults {
    SceneDefaults();

    const World World;
    const ViewCamera ViewCamera;
    const std::array<StudioLight, 4> StudioLights;
    const vec3 AmbientColor;
    const ViewportTheme ViewportTheme;
};
