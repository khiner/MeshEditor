#pragma once

#include "Camera.h"
#include "Lights.h"
#include "World.h"

#include "gpu/ViewportTheme.h"

struct SceneDefaults {
    SceneDefaults();

    const World World;
    const Camera Camera;
    const Lights Lights;
    const ViewportTheme ViewportTheme;
};
