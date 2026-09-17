#pragma once

#include "gpu/ViewportTheme.h"
#include "gpu/WorkspaceLights.h"
#include "render/LightComponents.h"
#include "viewport/ViewCamera.h"

#include <numbers>

constexpr float Pi{std::numbers::pi_v<float>};

void UpdateDerivedColors(ViewportTheme &);

struct Defaults {
    static constexpr vec3 WorldUp{0, 1, 0};
    static ViewCamera ViewCamera;
    static WorkspaceLights WorkspaceLights;
    static ViewportTheme ViewportTheme;
    static Perspective PerspectiveCamera;

    static constexpr float SpotOuterAngle{Pi / 4}, SpotBlend{0.15}, LightIntensity{100};
    static PunctualLight MakePunctualLight(PunctualLightType type) {
        return {
            .Range = 0.f, // infinite
            .Color = {1.f, 1.f, 1.f},
            .Intensity = LightIntensity,
            .InnerConeAngle = type == PunctualLightType::Spot ? SpotOuterAngle * (1.f - SpotBlend) : 0.f,
            .OuterConeAngle = type == PunctualLightType::Spot ? SpotOuterAngle : 0.f,
            .Type = type,
        };
    }
};
