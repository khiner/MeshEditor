#pragma once

#include "TypeList.h"
#include "numeric/uvec2.h"
#include "state/Entity.h"
#include "viewport/ViewportSchema.h"

struct WorkspaceLights;

// Whether the viewport's panel is open, and the view's logical size.
struct ViewportPanel {
    bool Open{true};
    numeric::uvec2 Extent{};
    bool operator==(const ViewportPanel &) const = default;
};

// The document's viewport that receives input.
struct ActiveViewport {
    state::Entity Viewport{state::Null};
    bool operator==(const ActiveViewport &) const = default;
};

// The value a new viewport starts with.
template<typename C> C ViewDefault() { return C{}; }
template<> WorkspaceLights ViewDefault<WorkspaceLights>();

// The workspace values a viewport entity holds, whose changes request its frame.
using ViewValues = TypeList<ViewportPanel, ViewCameraState, OrbitToActive, ViewportDisplay, MaterialPreviewLighting, RenderedLighting, WorkspaceLights, StudioEnvironment>;
