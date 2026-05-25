#pragma once

#include "gpu/InteractionMode.h"

#include <entt/entity/fwd.hpp>

// Viewport/camera-state operations: applied imperatively (not via PendingX components).

// Make `target` the look-through camera, preserving the saved view across switches.
void SetLookThrough(entt::registry &, entt::entity viewport, entt::entity target);
entt::entity LookThroughCameraEntity(const entt::registry &); // entt::null if none.

// Switches interaction mode, seeding/clearing per-mesh selection ranges as needed. Returns false if the switch is disallowed.
bool SetInteractionMode(entt::registry &, entt::entity viewport, InteractionMode);

// Prefilters (once) and activates the studio HDRI environment at `index`.
void SetStudioEnvironment(entt::registry &, uint32_t index);

// Emit the mode-appropriate delete/duplicate of the current selection.
void Delete(const entt::registry &, entt::entity viewport);
void Duplicate(const entt::registry &, entt::entity viewport);
