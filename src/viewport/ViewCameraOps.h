#pragma once

#include "viewport/ViewCamera.h"

#include <entt/entity/fwd.hpp>

#include <optional>

// Make `target` the look-through camera, preserving the saved view across switches. No-op if `target` is not a camera.
void SetLookThrough(entt::registry &, entt::entity viewport, entt::entity target);
// Exit look-through, restoring the saved pre-look-through view camera. No-op if not looking through.
void ClearLookThrough(entt::registry &, entt::entity viewport);
entt::entity LookThroughCameraEntity(const entt::registry &); // entt::null if none.

// The active view camera plus any look-through camera's saved view.
// Replay doesn't record navigation, so this is captured before a clear/replay and restored afterward.
struct ViewCameraState {
    ViewCamera Active;
    std::optional<ViewCamera> LookThroughSaved;
};
ViewCameraState GetViewCameraState(const entt::registry &, entt::entity viewport);
void SetViewCameraState(entt::registry &, entt::entity viewport, ViewCameraState);
