#pragma once

#include "Camera.h"
#include "SceneComponents.h"
#include "SceneTransformUtils.h"
#include "TransformGizmo.h"
#include "gpu/Element.h"
#include "gpu/InteractionMode.h"
#include "numeric/quat.h"
#include "numeric/vec3.h"

#include <entt/entity/fwd.hpp>

#include <optional>

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
// Handler snapshots `StartTransform`/`StartBoneLength` for any affected entity that doesn't yet have one.
struct DragGizmo {
    std::vector<std::pair<entt::entity, Transform>> Locals;
    std::vector<std::pair<entt::entity, float>> BoneDisplayScales;
};
struct DragGizmoMeshEdit {
    std::unique_ptr<PendingTransform> Value;
};
struct EndGizmoDrag {};

// Toolbar viewport-tool selection. Tools are mutually exclusive: picking a select tool clears the
// transform type; picking a transform tool keeps the selection gesture but suppresses it visually.
struct SetActiveTool {
    enum class Tool : uint8_t { SelectBox,
                                SelectClick,
                                Translate,
                                Rotate,
                                Scale,
                                Universal };
    Tool Value;
};

// Latched transform-type for the next gizmo drag. `nullopt` clears the latch (consumed by InteractOverlay).
struct SetStartScreenTransform {
    std::optional<TransformGizmo::TransformType> Value;
};

using Actions = std::variant<
    SetInteractionMode, CycleInteractionMode, SetEditMode,
    EnterLookThroughCamera, ExitLookThroughCamera, Play,
    SetViewportShading, SelectAll, OrbitViewCamera, ZoomViewCamera,
    ApplyExciteImpact, ClearExciteImpacts,
    SetStudioEnvironment, SetSourceIblIntensity,
    ResetViewCamera, ResetViewportTheme, ResetPbrLighting,
    SetViewCameraTarget, SetViewCameraLens, SetViewCameraTargetDirection, SetPbrMeshFeaturesMask,
    SetRotationUiMode, SetTransformRotationFromUi,
    DragGizmo, DragGizmoMeshEdit, EndGizmoDrag, SetActiveTool, SetStartScreenTransform>;
} // namespace action::scene
