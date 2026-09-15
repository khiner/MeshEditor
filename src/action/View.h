#pragma once

#include "CameraTypes.h"
#include "action/Core.h"
#include "gizmo/TransformGizmoTypes.h"
#include "gpu/Element.h"
#include "gpu/InteractionMode.h"
#include "gpu/WorkspaceLights.h"
#include "scene/RotationUi.h"
#include "viewport/ViewportDisplay.h"

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
struct SetLookThroughCamera {
    state::Entity Entity;
};
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
// Replaces the viewport's solid-mode lights. Shared ownership bounds the action variant size.
struct SetWorkspaceLights {
    std::unique_ptr<WorkspaceLights> Value;
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
    Scope Scope{Scope::Active};
};
// `R` must already be normalized. Targets the active bone in Pose mode, otherwise the active entity.
struct SetTransformRotationFromUi {
    quat R;
    RotationUiVariant UiVariant;
    Scope Scope{Scope::Active};
};
struct DragGizmo {
    std::unique_ptr<PendingTransform> Value;
};
struct DragGizmoMeshEdit {
    std::unique_ptr<PendingTransform> Value;
};
struct EndGizmoDrag {};

// Select tools clear the transform type, while transform tools retain the hidden selection gesture.
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

// Latches a transform type for the next gizmo drag and restores any active drag to its initial state.
// This live-only action aborts the staged gesture and is excluded from recording.
struct LatchScreenTransform {
    TransformGizmo::TransformType Value;
};
// Clear the screen-transform latch once consumed by InteractOverlay. Live-only bookkeeping, not recorded.
struct ClearScreenTransformLatch {};

// Logical (window) size of the viewport.
// Apply only sets the ViewportExtent ctx value, the GPU resize happens later.
struct SetExtent {
    uvec2 Extent;
};

// Set the viewport's studio HDRI / image-based lighting environment.
// Identified by source HDRI name (not the directory-scan index) so it stays stable across runs.
struct SetStudioEnvironment {
    std::string Name;
};
struct SetSourceIblIntensity {
    float Intensity;
};

// Make `Scene` the active scene shown in the viewport.
struct SetActiveScene {
    state::Entity Scene;
};

using Action = std::variant<
    SetInteractionMode, CycleInteractionMode, SetEditMode,
    EnterLookThroughCamera, ExitLookThroughCamera, SetLookThroughCamera,
    SetViewportShading, OrbitViewCamera, ZoomViewCamera,
    ResetViewCamera, ResetViewportTheme, ResetPbrLighting, SetWorkspaceLights,
    SetViewCameraTarget, SetViewCameraLens, SetViewCameraTargetDirection,
    SetRotationUiMode, SetTransformRotationFromUi,
    DragGizmo, DragGizmoMeshEdit, EndGizmoDrag, SetActiveTool, LatchScreenTransform, ClearScreenTransformLatch,
    SetExtent, SetStudioEnvironment, SetSourceIblIntensity, SetActiveScene>;

void Apply(state::Scene &, state::Entity viewport, const Action &);
} // namespace action::view
