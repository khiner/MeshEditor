#pragma once

#include "TransformGizmoTypes.h"
#include "TransformUtils.h"
#include "Variant.h"
#include "ViewportComponents.h"
#include "action/Core.h"

struct PendingTransform;

namespace action::view {
struct SetInteractionMode {
    InteractionMode Mode;
};
struct CycleInteractionMode {};
struct SetEditMode {
    Element Mode;
};
struct EnterLookThroughCamera {};
struct ExitLookThroughCamera {};
struct SetViewportShading {
    ViewportShadingMode Mode;
};
struct OrbitViewCamera {
    vec2 DeltaRad;
};
struct ZoomViewCamera {
    float Factor;
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
    enum class Tool : uint8_t {
        SelectBox,
        SelectClick,
        Translate,
        Rotate,
        Scale,
        Universal
    };
    Tool Value;
};

// Latched transform-type for the next gizmo drag. `nullopt` clears the latch (consumed by InteractOverlay).
struct SetStartScreenTransform {
    std::optional<TransformGizmo::TransformType> Value;
};

// Studio HDRI / image-based lighting environment for the viewport.
struct SetStudioEnvironment {
    uint32_t Index;
};
struct SetSourceIblIntensity {
    float Intensity;
};

using Actions = std::variant<
    SetInteractionMode, CycleInteractionMode, SetEditMode,
    EnterLookThroughCamera, ExitLookThroughCamera,
    SetViewportShading, OrbitViewCamera, ZoomViewCamera,
    ResetViewCamera, ResetViewportTheme, ResetPbrLighting,
    SetViewCameraTarget, SetViewCameraLens, SetViewCameraTargetDirection,
    SetRotationUiMode, SetTransformRotationFromUi,
    DragGizmo, DragGizmoMeshEdit, EndGizmoDrag, SetActiveTool, SetStartScreenTransform,
    SetStudioEnvironment, SetSourceIblIntensity>;

using Action = MergedVariantT<
    Actions,
    Replace<::Camera>, ReplaceActive<::Camera>,
    Update<TransformGizmo::Type>, Update<TransformGizmo::Mode>,
    Update<DebugChannel>, Update<vk::ClearColorValue>>;

void Apply(entt::registry &, entt::entity viewport, const Action &);
} // namespace action::view
