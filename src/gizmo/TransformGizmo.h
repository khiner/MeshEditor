#pragma once

#include <optional>

#include "gizmo/TransformGizmoTypes.h"
#include "numeric/rect.h"

struct ViewCamera;
struct GizmoInteraction;

namespace colors {
struct AxesArray;
} // namespace colors

namespace TransformGizmo {
std::string_view ToString(const GizmoInteraction &);

struct Result {
    Transform Start; // Transform at interaction start
    Transform Delta; // Delta transform since interaction start
};

// Processes interaction (hover, click, drag) and returns the transform delta if actively dragging.
// Does NOT render — call Render() afterward with the (potentially updated) transform stored in RenderTransform.
// The overlay of the active viewport sets RenderTransform each frame, and every displayed viewport renders it.
std::optional<Result> Interact(GizmoInteraction &, const GizmoTransform &, Config, const ViewCamera &, rect viewport, vec2 mouse_px, std::optional<TransformType> start_screen_transform = {});

// Renders the gizmo at RenderTransform from the camera, with the highlight and guides of the interaction.
void Render(const GizmoInteraction &, Type, const ViewCamera &, rect viewport, const colors::AxesArray &);
} // namespace TransformGizmo
