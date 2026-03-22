#include "SceneDefaults.h"

namespace {
vec3 Shade(vec3 c, int offset) { return glm::clamp(c + float(offset) / 255.f, 0.f, 1.f); }
vec3 BlendShade(vec3 c1, vec3 c2, float fac, int offset = 0) { return glm::clamp(glm::mix(c1, c2, fac) + float(offset) / 255.f, 0.f, 1.f); }

constexpr vec3 RawAxisX{1.f, 51.f / 255.f, 82.f / 255.f};
constexpr vec3 RawAxisZ{40.f / 255.f, 144.f / 255.f, 1.f};
} // namespace

void UpdateDerivedColors(ViewportThemeColors &c) {
    // Grid-derived colors, matching Blender's overlay_instance.cc
    c.GridLine = {Shade(vec3(c.Grid), 10), c.Grid.w};
    c.GridEmphasis = {Shade(vec3(c.Grid), 20), 1.f};
    c.GridAxisX = BlendShade(vec3(c.Grid), RawAxisX, 0.85f, -20);
    c.GridAxisZ = BlendShade(vec3(c.Grid), RawAxisZ, 0.85f, -20);

    c.BoneActive = Shade(c.EdgeSelectedIncidental, 60);
    c.BoneActiveUnsel = BlendShade(c.WireEdit, c.EdgeSelectedIncidental, 0.15f);
    c.BoneSelect = Shade(c.EdgeSelectedIncidental, -20);
    c.BonePoseActiveUnsel = BlendShade(c.Wire, c.BonePose, 0.15f);
}

World SceneDefaults::World{.Origin{0, 0, 0}, .Up{0, 1, 0}};
ViewCamera SceneDefaults::ViewCamera{
    {8.189f, 4.458f, 3.616f},
    {0, 0, 0},
    {Perspective{.FieldOfViewRad = 2.f * std::atan(0.72f * 9.f / 16.f), .FarClip = 1000.f, .NearClip = 0.01f}},
};
// Blender's default BKE_studiolight_default values (studiolight.cc)
WorkspaceLights SceneDefaults::WorkspaceLights{
    .Lights = {{
        {.Direction = {-0.352546f, 0.170931f, -0.920051f}, .SpecularColor = {0.266761f, 0.266761f, 0.266761f}, .DiffuseColor = {0.033103f, 0.033103f, 0.033103f}, .Wrap = 0.526620f},
        {.Direction = {-0.408163f, 0.346939f, 0.844415f}, .SpecularColor = {0.599030f, 0.599030f, 0.599030f}, .DiffuseColor = {0.521083f, 0.538226f, 0.538226f}, .Wrap = 0.0f},
        {.Direction = {0.521739f, 0.826087f, 0.212999f}, .SpecularColor = {0.106102f, 0.125981f, 0.158523f}, .DiffuseColor = {0.038403f, 0.034357f, 0.049530f}, .Wrap = 0.478261f},
        {.Direction = {0.624519f, -0.562067f, -0.542269f}, .SpecularColor = {0.106535f, 0.084771f, 0.066080f}, .DiffuseColor = {0.090838f, 0.082080f, 0.072255f}, .Wrap = 0.200000f},
    }},
    .AmbientColor = {0, 0, 0},
    .UseSpecular = 1,
};
static ViewportTheme MakeDefaultViewportTheme() {
    ViewportTheme theme{
        .Colors{
            .Grid{0.329f, 0.329f, 0.329f, 0.502f},
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
