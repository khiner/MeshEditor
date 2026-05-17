#pragma once

#include "Camera.h"
#include "gpu/Element.h"
#include "gpu/InteractionMode.h"
#include "numeric/quat.h"
#include "numeric/vec3.h"
#include "scene_impl/SceneComponents.h"
#include "scene_impl/SceneTransformUtils.h"

#include <entt/entity/fwd.hpp>

struct PendingTransform;

namespace action::scene {
struct SetInteractionMode {
    InteractionMode Mode;
};
struct CycleInteractionMode {};
struct SetEditMode {
    Element Mode;
};
struct EnterLookThroughCamera {};
struct ExitLookThroughCamera {};
struct Play {};
struct SetViewportShading {
    ViewportShadingMode Mode;
};
struct SelectAll {};
struct OrbitViewCamera {
    vec2 DeltaRad;
};
struct ZoomViewCamera {
    float Factor;
};
struct ApplyExciteImpact {
    entt::entity InstanceEntity;
    uint32_t VertexIndex;
};
struct ClearExciteImpacts {};
struct SetStudioEnvironment {
    uint32_t Index;
};
struct SetSourceIblIntensity {
    float Intensity;
};
struct ResetViewCamera {};
struct ResetViewportTheme {};
struct ResetPbrLighting {
    bool Rendered;
};
struct SetViewCameraTarget {
    vec3 Target;
};
struct SetViewCameraLens {
    ::Camera Data;
};
struct SetViewCameraTargetDirection {
    vec3 Direction;
};
// `Mask=0` removes the component. Targets the active mesh entity.
struct SetPbrMeshFeaturesMask {
    uint32_t Mask;
};
// Targets the active bone in Pose mode, otherwise the active entity.
struct SetRotationUiMode {
    int Index;
};
// `R` must already be normalized. Targets the active bone in Pose mode, otherwise the active entity.
struct SetTransformRotationFromUi {
    quat R;
    RotationUiVariant UiVariant;
};
struct BeginGizmoDrag {
    std::vector<std::pair<entt::entity, StartTransform>> Starts;
    std::vector<std::pair<entt::entity, float>> StartBoneLengths;
};
struct UpdateGizmoDragLocals {
    std::vector<std::pair<entt::entity, Transform>> Locals;
    std::vector<std::pair<entt::entity, float>> BoneDisplayScales;
};
struct UpdateGizmoMeshEditPending {
    std::unique_ptr<PendingTransform> Value;
};
struct EndGizmoDrag {};

using Actions = std::variant<
    SetInteractionMode, CycleInteractionMode, SetEditMode,
    EnterLookThroughCamera, ExitLookThroughCamera, Play,
    SetViewportShading, SelectAll, OrbitViewCamera, ZoomViewCamera,
    ApplyExciteImpact, ClearExciteImpacts,
    SetStudioEnvironment, SetSourceIblIntensity,
    ResetViewCamera, ResetViewportTheme, ResetPbrLighting,
    SetViewCameraTarget, SetViewCameraLens, SetViewCameraTargetDirection, SetPbrMeshFeaturesMask,
    SetRotationUiMode, SetTransformRotationFromUi,
    BeginGizmoDrag, UpdateGizmoDragLocals, UpdateGizmoMeshEditPending, EndGizmoDrag>;
} // namespace action::scene
