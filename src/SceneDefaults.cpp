#include "SceneDefaults.h"

#include <glm/common.hpp>

namespace {
vec3 Shade(vec3 c, int offset) { return glm::clamp(c + float(offset) / 255.f, 0.f, 1.f); }
vec3 BlendShade(vec3 c1, vec3 c2, float fac, int offset = 0) { return glm::clamp(glm::mix(c1, c2, fac) + float(offset) / 255.f, 0.f, 1.f); }
} // namespace

void UpdateDerivedColors(ViewportThemeColors &c) {
    c.BoneActive = Shade(c.EdgeSelectedIncidental, 60);
    c.BoneActiveUnsel = BlendShade(c.WireEdit, c.EdgeSelectedIncidental, 0.15f);
    c.BoneSelect = Shade(c.EdgeSelectedIncidental, -20);
    c.BonePoseActiveUnsel = BlendShade(c.Wire, c.BonePose, 0.15f);
}

World SceneDefaults::World{.Origin{0, 0, 0}, .Up{0, 1, 0}};
ViewCamera SceneDefaults::ViewCamera{{0, 0, 2}, {0, 0, 0}, {DefaultPerspectiveCamera()}};
WorkspaceLights SceneDefaults::WorkspaceLights{
    .ViewColor = {1, 1, 1},
    .AmbientIntensity = 0.1f,
    .DirectionalColor = {1, 1, 1},
    .DirectionalIntensity = 0.15f,
    .Direction = {-1, -1, -1},
};
static ViewportTheme MakeDefaultViewportTheme() {
    ViewportTheme theme{
        .Colors{
            .Wire{0, 0, 0},
            .WireEdit{0, 0, 0},
            .ObjectActive{1, 0.627f, 0.157f},
            .ObjectSelected{0.929f, 0.341f, 0},
            .Light{0, 0, 0, 0.314f},
            .Vertex{0, 0, 0},
            .VertexSelected{1, 0.478f, 0},
            .EdgeSelectedIncidental{1, 0.6f, 0},
            .EdgeSelected{1, 0.847f, 0},
            .FaceSelectedIncidental{1, 0.639f, 0, 0.2f},
            .FaceSelected{1, 0.718f, 0, 0.2f},
            .ElementActive{1, 1, 1, 0.2f},
            .FaceNormal{0.133f, 0.867f, 0.867f},
            .VertexNormal{0.137f, 0.380f, 0.867f},
            .BoneSolid{0.698f, 0.698f, 0.698f},
            .BonePose{0.314f, 0.784f, 1.0f},
            .BonePoseActive{0.549f, 1.0f, 1.0f},
            .Transform{1, 1, 1},
        },
        .EdgeWidth = 0.5f, // Half-width in pixels
        .SilhouetteEdgeWidth = 2,
    };
    UpdateDerivedColors(theme.Colors);
    return theme;
}

ViewportTheme SceneDefaults::ViewportTheme = MakeDefaultViewportTheme();

PunctualLight SceneDefaults::MakePunctualLight(PunctualLightType type) {
    return {
        .Range = type == PunctualLightType::Directional ? 0.f : PointRange,
        .Color = {1.f, 1.f, 1.f},
        .Intensity = LightIntensity,
        .InnerConeCos = type == PunctualLightType::Spot ? std::cos(SpotOuterAngle * (1.f - SpotBlend)) : 0.f,
        .OuterConeCos = type == PunctualLightType::Spot ? std::cos(SpotOuterAngle) : 0.f,
        .Type = type,
    };
}
