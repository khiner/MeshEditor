#pragma once

#include "scene/Defaults.h"
#include "state/Entity.h"
#include "viewport/ViewportSchema.h"

// The viewport's active editor view, the scene camera it looks through or null, and the view it saved on entering look-through.
// The view keeps its pose while the camera it looks through is absent, and follows it again when it returns.
struct ViewCameraState {
    ViewCamera Active{Defaults::ViewCamera};
    state::Entity LookThrough{state::Null};
    std::optional<ViewCamera> LookThroughSaved;

    bool operator==(const ViewCameraState &) const = default;
};

// Make `target` the look-through camera, preserving the saved view across switches.
// No-op if `target` is not a camera.
void SetLookThrough(const state::Registry &, ViewCameraState &, state::Entity target);
// Exit look-through, restoring the saved pre-look-through view camera.
// No-op if not looking through.
void ClearLookThrough(ViewCameraState &);
// The camera the viewport looks through, or null while that camera is absent from the viewport's scene.
state::Entity LookThroughCamera(const state::Registry &, state::Entity viewport);
