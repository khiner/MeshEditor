#include "SceneDefaults.h"

World SceneDefaults::World{.Origin{0, 0, 0}, .Up{0, 1, 0}};

ViewCamera SceneDefaults::ViewCamera{
    {0, 0, 2},
    {0, 0, 0},
    {DefaultPerspectiveCamera()},
};
WorkspaceLights SceneDefaults::WorkspaceLights{
    .ViewColor = {1, 1, 1},
    .AmbientIntensity = 0.1f,
    .DirectionalColor = {1, 1, 1},
    .DirectionalIntensity = 0.15f,
    .Direction = {-1, -1, -1},
};
ViewportTheme SceneDefaults::ViewportTheme{
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
        .Transform{1, 1, 1},
        .BoneSolid{0.698f, 0.698f, 0.698f},
        .BoneActive{1.0f, 0.835f, 0.235f},
        .BoneActiveUnsel{0.149f, 0.086f, 0.0f},
        .BoneSelect{0.922f, 0.522f, 0.0f},
        .BonePose{0.314f, 0.784f, 1.0f},
        .BonePoseActive{0.549f, 1.0f, 1.0f},
        .BonePoseActiveUnsel{0.047f, 0.118f, 0.149f},
    },
    .EdgeWidth = 1.0f,
    .SilhouetteEdgeWidth = 2,
};

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
