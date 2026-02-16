#pragma once

#include "Lights.h"
#include "ViewCamera.h"
#include "World.h"

#include "gpu/ViewportTheme.h"

struct SceneDefaults {
    SceneDefaults();

    const World World;
    const ViewCamera ViewCamera;
    const Lights Lights;
    const ViewportTheme ViewportTheme;
};
