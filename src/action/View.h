#pragma once

#include "numeric/uvec2.h"
#include "numeric/vec2.h"

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
// Flips the X-ray flag of the current shading family.
struct ToggleXRay {};
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
    CameraLens Data;
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
// Applies a pivot and delta to the selected objects or bones.
struct TransformSelection {
    std::unique_ptr<PendingTransform> Value;
};
// Stages a pivot and delta over the edit-mode elements, which move on commit.
struct TransformElements {
    std::unique_ptr<PendingTransform> Value;
};
// Clears the transform start state and the latch.
struct EndTransform {};

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

// Latches a transform type for the next drag and restores any active drag to its initial state.
// This live-only action aborts the staged gesture and is excluded from recording.
struct LatchTransform {
    TransformGizmo::TransformType Value;
};

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

// Make `Scene` the active scene shown in the viewport.
struct SetActiveScene {
    state::Entity Scene;
};

using Action = std::variant<
    SetInteractionMode, CycleInteractionMode, SetEditMode,
    EnterLookThroughCamera, ExitLookThroughCamera, SetLookThroughCamera,
    SetViewportShading, ToggleXRay, OrbitViewCamera, ZoomViewCamera,
    ResetViewCamera, ResetViewportTheme, ResetPbrLighting, SetWorkspaceLights,
    SetViewCameraTarget, SetViewCameraLens, SetViewCameraTargetDirection,
    SetRotationUiMode, SetTransformRotationFromUi,
    TransformSelection, TransformElements, EndTransform, SetActiveTool, LatchTransform,
    SetExtent, SetStudioEnvironment, SetActiveScene>;

void Apply(state::Scene &, state::Entity viewport, const Action &);
} // namespace action::view
