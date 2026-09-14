#pragma once

#include "viewport/ViewCamera.h"

#include "state/Entity.h"

#include <optional>

// Make `target` the look-through camera, preserving the saved view across switches. No-op if `target` is not a camera.
void SetLookThrough(state::Scene &, state::Entity viewport, state::Entity target);
// Exit look-through, restoring the saved pre-look-through view camera. No-op if not looking through.
void ClearLookThrough(state::Scene &, state::Entity viewport);
// Returns null when no look-through camera is active.
state::Entity LookThroughCameraEntity(const state::Scene &);

// The active view camera plus any look-through camera's saved view.
// Replay doesn't record navigation, so this is captured before a clear/replay and restored afterward.
struct ViewCameraState {
    ViewCamera Active;
    std::optional<ViewCamera> LookThroughSaved;
};
ViewCameraState GetViewCameraState(const state::Scene &, state::Entity viewport);
void SetViewCameraState(state::Scene &, state::Entity viewport, ViewCameraState);
