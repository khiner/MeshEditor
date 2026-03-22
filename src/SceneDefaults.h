#pragma once

#include "ViewCamera.h"
#include "World.h"
#include "gpu/PunctualLight.h"
#include "gpu/ViewportTheme.h"
#include "gpu/WorkspaceLights.h"

#include <numbers>

constexpr float Pi{std::numbers::pi_v<float>};

void UpdateDerivedColors(ViewportThemeColors &);

struct SceneDefaults {
    static World World;
    static ViewCamera ViewCamera;
    static WorkspaceLights WorkspaceLights;
    static ViewportTheme ViewportTheme;

    static constexpr float SpotOuterAngle{Pi / 4}, SpotBlend{0.15}, LightIntensity{75}, PointRange{10};
    static PunctualLight MakePunctualLight(PunctualLightType = PunctualLightType::Point);
};
