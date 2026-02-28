#pragma once

#include "ViewCamera.h"
#include "World.h"
#include "gpu/PunctualLight.h"
#include "gpu/ViewportTheme.h"
#include "gpu/WorkspaceLights.h"

#include <numbers>

constexpr float Pi{std::numbers::pi_v<float>};

struct SceneDefaults {
    SceneDefaults();

    const World World;
    const ViewCamera ViewCamera;
    const WorkspaceLights WorkspaceLights;
    const ViewportTheme ViewportTheme;

    static PunctualLight MakePunctualLight(PunctualLightType = PunctualLightType::Point);
    static constexpr float SpotOuterAngle{Pi / 4}, SpotBlend{0.15}, LightIntensity{75}, PointRange{10};
};
