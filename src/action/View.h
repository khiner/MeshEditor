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
// The view actions name the viewport they change by its Id.
struct EnterLookThroughCamera {
    uint32_t ViewportId;
};
struct ExitLookThroughCamera {
    uint32_t ViewportId;
};
struct SetLookThroughCamera {
    uint32_t ViewportId;
    state::Entity Entity;
};
struct SetViewportShading {
    uint32_t ViewportId;
    ViewportShadingMode Mode;
};
// Flips the X-ray flag of the current shading family.
struct ToggleXRay {
    uint32_t ViewportId;
};
struct OrbitViewCamera {
    uint32_t ViewportId;
    vec2 DeltaRad;
};
struct ZoomViewCamera {
    uint32_t ViewportId;
    float Factor;
};
struct ResetViewCamera {
    uint32_t ViewportId;
};
struct ResetViewportTheme {};
struct ResetNodeTheme {};
struct ResetInterfaceTheme {};
struct ResetTimelineTheme {};
struct ResetPbrLighting {
    uint32_t ViewportId;
    bool Rendered;
};
// Replaces the viewport's solid-mode lights.
// Shared ownership bounds the action variant size.
struct SetWorkspaceLights {
    uint32_t ViewportId;
    std::unique_ptr<WorkspaceLights> Value;
};
struct SetViewCameraTarget {
    uint32_t ViewportId;
    vec3 Target;
};
struct SetViewCameraLens {
    uint32_t ViewportId;
    CameraLens Data;
};
struct SetViewCameraTargetDirection {
    uint32_t ViewportId;
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
    uint32_t ViewportId;
    std::string Name;
};

// Make `Scene` the active scene shown in the viewport.
struct SetActiveScene {
    state::Entity Scene;
};

// Opens a viewport with the default view values.
struct AddViewport {};
// Closes the viewport with the Id, unless it is the only one.
struct RemoveViewport {
    uint32_t ViewportId;
};

using Action = std::variant<
    SetInteractionMode, CycleInteractionMode, SetEditMode,
    EnterLookThroughCamera, ExitLookThroughCamera, SetLookThroughCamera,
    SetViewportShading, ToggleXRay, OrbitViewCamera, ZoomViewCamera,
    ResetViewCamera, ResetViewportTheme, ResetNodeTheme, ResetInterfaceTheme, ResetTimelineTheme, ResetPbrLighting, SetWorkspaceLights,
    SetViewCameraTarget, SetViewCameraLens, SetViewCameraTargetDirection,
    TransformSelection, TransformElements, SetActiveTool, LatchTransform,
    SetStudioEnvironment, SetActiveScene, AddViewport, RemoveViewport>;

void Apply(state::Registry &, const Action &);
} // namespace action::view
