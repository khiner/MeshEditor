#pragma once

#include "CameraTypes.h"
#include "gizmo/TransformGizmoTypes.h"
#include "gpu/Element.h"
#include "gpu/WorkspaceLights.h"
#include "viewport/ViewportDisplay.h"

namespace action::view {
struct SetInteractionMode {
    InteractionMode Mode;
};
struct CycleInteractionMode {};
struct SetEditMode {
    Element Mode;
};
// The view actions name the viewport entity they change.
struct EnterLookThroughCamera {
    state::Entity Viewport;
};
struct ExitLookThroughCamera {
    state::Entity Viewport;
};
struct SetLookThroughCamera {
    state::Entity Viewport;
    state::Entity Entity;
};
// Shows `Scene` in the viewport.
struct SetViewportScene {
    state::Entity Viewport;
    state::Entity Scene;
};
struct SetViewportShading {
    state::Entity Viewport;
    ViewportShadingMode Mode;
};
// Flips the X-ray flag of the current shading family.
struct ToggleXRay {
    state::Entity Viewport;
};
struct OrbitViewCamera {
    state::Entity Viewport;
    vec2 DeltaRad;
};
struct ZoomViewCamera {
    state::Entity Viewport;
    float Factor;
};
struct ResetViewCamera {
    state::Entity Viewport;
};
struct ResetViewportTheme {};
struct ResetNodeTheme {};
struct ResetInterfaceTheme {};
struct ResetTimelineTheme {};
struct ResetPbrLighting {
    state::Entity Viewport;
    bool Rendered;
};
// Replaces the viewport's solid-mode lights.
// Shared ownership bounds the action variant size.
struct SetWorkspaceLights {
    state::Entity Viewport;
    std::unique_ptr<WorkspaceLights> Value;
};
struct SetViewCameraTarget {
    state::Entity Viewport;
    vec3 Target;
};
struct SetViewCameraLens {
    state::Entity Viewport;
    CameraLens Data;
};
struct SetViewCameraTargetDirection {
    state::Entity Viewport;
    vec3 Direction;
};
// Applies a delta about the selection's pivot to the selected objects or bones.
struct TransformSelection {
    Transform Delta;
};
// Stages a delta about the selection's pivot over the edit-mode elements, which move on commit.
struct TransformElements {
    Transform Delta;
};

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

// Set the viewport's studio HDRI / image-based lighting environment.
// Identified by source HDRI name (not the directory-scan index) so it stays stable across runs.
struct SetStudioEnvironment {
    state::Entity Viewport;
    std::string Name;
};

// Opens a viewport showing the frame's scene.
struct AddViewport {};
// Closes the viewport, unless it is the only one.
struct RemoveViewport {
    state::Entity Viewport;
};

using Action = std::variant<
    SetInteractionMode, CycleInteractionMode, SetEditMode,
    EnterLookThroughCamera, ExitLookThroughCamera, SetLookThroughCamera, SetViewportScene,
    SetViewportShading, ToggleXRay, OrbitViewCamera, ZoomViewCamera,
    ResetViewCamera, ResetViewportTheme, ResetNodeTheme, ResetInterfaceTheme, ResetTimelineTheme, ResetPbrLighting, SetWorkspaceLights,
    SetViewCameraTarget, SetViewCameraLens, SetViewCameraTargetDirection,
    TransformSelection, TransformElements, SetActiveTool, LatchTransform,
    SetStudioEnvironment, AddViewport, RemoveViewport>;

void Apply(state::Registry &, const Action &);
} // namespace action::view
