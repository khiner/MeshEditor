#pragma once

#include "TransformGizmoTypes.h"
#include "numeric/rect.h"

#include <string_view>

struct ViewCamera;
struct GizmoInteraction;

namespace colors {
struct AxesArray;
}

namespace TransformGizmo {
std::string_view ToString(const GizmoInteraction &);

struct Result {
    Transform Start; // Transform at interaction start
    Transform Delta; // Delta transform since interaction start
};

// Processes interaction (hover, click, drag) and returns the transform delta if actively dragging.
// Does NOT render — call Render() afterward with the (potentially updated) transform stored in RenderTransform.
std::optional<Result> Interact(GizmoInteraction &, const GizmoTransform &, Config, const ViewCamera &, rect viewport, vec2 mouse_px, std::optional<TransformType> start_screen_transform = {});

// Renders the gizmo (RenderTransform + Delta from the last Interact() call), then clears RenderTransform.
void Render(GizmoInteraction &, Type, const ViewCamera &, rect viewport, const colors::AxesArray &);
} // namespace TransformGizmo
