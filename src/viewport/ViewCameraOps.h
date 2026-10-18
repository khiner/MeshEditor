#pragma once

#include "viewport/ViewCamera.h"

#include "state/Entity.h"

// Make `target` the look-through camera, preserving the saved view across switches.
// No-op if `target` is not a camera.
void SetLookThrough(state::Registry &, state::Entity viewport, state::Entity target);
// Exit look-through, restoring the saved pre-look-through view camera.
// No-op if not looking through.
void ClearLookThrough(state::Registry &, state::Entity viewport);
// Returns null when no look-through camera is active.
state::Entity LookThroughCameraEntity(const state::Registry &);

// The viewport's active and saved editor views, owned by the workspace.
struct ViewCameraState {
    ViewCamera Active;
    std::optional<ViewCamera> LookThroughSaved;
};
ViewCameraState GetViewCameraState(const state::Registry &, state::Entity viewport);
void SetViewCameraState(state::Registry &, state::Entity viewport, ViewCameraState);
