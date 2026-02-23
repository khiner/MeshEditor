#pragma once

#include "ViewCamera.h"
#include "World.h"

#include "gpu/ViewportTheme.h"
#include "gpu/WorkspaceLights.h"

struct SceneDefaults {
    SceneDefaults();

    const World World;
    const ViewCamera ViewCamera;
    const WorkspaceLights WorkspaceLights;
    const ViewportTheme ViewportTheme;
};
