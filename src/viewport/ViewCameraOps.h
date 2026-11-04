#pragma once

#include "scene/Defaults.h"
#include "state/Entity.h"
#include "viewport/ViewportSchema.h"

// The viewport's active editor view, the scene camera it looks through or null, and the view it saved on entering look-through.
struct ViewCameraState {
    ViewCamera Active{Defaults::ViewCamera};
    state::Entity LookThrough{state::Null};
    std::optional<ViewCamera> LookThroughSaved;

    bool operator==(const ViewCameraState &) const = default;
};

// Makes `target` the look-through camera and keeps the saved view across switches.
// Does nothing when `target` is not a camera.
void SetLookThrough(const state::Registry &, ViewCameraState &, state::Entity target);
// Exits look-through and restores the saved view.
void ClearLookThrough(ViewCameraState &);
void LeaveLookThrough(ViewCameraState &, const ViewCamera &shown);
// The camera the viewport looks through, or null while that camera is absent from the viewport's scene.
state::Entity LookThroughCamera(const state::Registry &, state::Entity viewport);
// A viewport looking through a camera views from the camera's pose through its lens widened to the viewport's aspect, once its transition into the camera ends.
ViewCamera ViewCameraOf(const state::Registry &, state::Entity viewport);
