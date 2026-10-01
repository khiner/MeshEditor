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

// The viewport's active and saved editor views, owned by the workspace.
struct ViewCameraState {
    ViewCamera Active;
    std::optional<ViewCamera> LookThroughSaved;
};
ViewCameraState GetViewCameraState(const state::Scene &, state::Entity viewport);
void SetViewCameraState(state::Scene &, state::Entity viewport, ViewCameraState);
