#pragma once

#include "viewport/ViewCamera.h"

#include "state/Entity.h"

// The viewport's active and saved editor views, owned by the workspace.
struct ViewCameraState {
    ViewCamera Active;
    std::optional<ViewCamera> LookThroughSaved;

    bool operator==(const ViewCameraState &) const = default;
};

struct ViewportState;

// Make `target` the look-through camera, preserving the saved view across switches.
// No-op if `target` is not a camera.
void SetLookThrough(const state::Registry &, ViewportState &, state::Entity target);
// Exit look-through, restoring the saved pre-look-through view camera.
// No-op if not looking through.
void ClearLookThrough(ViewportState &);
