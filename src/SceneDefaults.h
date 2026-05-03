#pragma once

#include "ViewCamera.h"
#include "World.h"
#include "gpu/PunctualLight.h"
#include "gpu/ViewportTheme.h"
#include "gpu/WorkspaceLights.h"

#include <numbers>

constexpr float Pi{std::numbers::pi_v<float>};

void UpdateDerivedColors(ViewportTheme &);

struct SceneDefaults {
    static World World;
    static ViewCamera ViewCamera;
    static WorkspaceLights WorkspaceLights;
    static ViewportTheme ViewportTheme;

    static constexpr float SpotOuterAngle{Pi / 4}, SpotBlend{0.15}, LightIntensity{75}, PointRange{10};
    static PunctualLight MakePunctualLight(PunctualLightType type) {
        return {
            .Range = type == PunctualLightType::Directional ? 0.f : PointRange,
            .Color = {1.f, 1.f, 1.f},
            .Intensity = LightIntensity,
            .InnerConeCos = type == PunctualLightType::Spot ? std::cos(SpotOuterAngle * (1.f - SpotBlend)) : 0.f,
            .OuterConeCos = type == PunctualLightType::Spot ? std::cos(SpotOuterAngle) : 0.f,
            .Type = type,
        };
    }
};
