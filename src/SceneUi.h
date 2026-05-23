#pragma once

#include "Action.h"

#include <entt/entity/fwd.hpp>

struct SceneFrameState;

// Mouse/keyboard interaction for the viewport. Mutates the per-frame Frame state.
void Interact(entt::registry &, entt::entity scene_entity, SceneFrameState &Frame, action::Emit);
// Overlay controls (gizmos, mode pickers). Mutates Frame.
void InteractOverlay(entt::registry &, entt::entity scene_entity, SceneFrameState &Frame, action::Emit);
// Reads Frame state set by InteractOverlay and renders the overlay (gizmo, box-select rectangle, origin dots).
void DrawOverlay(entt::registry &, entt::entity scene_entity, SceneFrameState &Frame);
// Side-panel controls (interaction mode, object tree, materials, etc.).
void RenderControls(entt::registry &, entt::entity scene_entity, action::Emit);

void RenderClipPickers(entt::registry &, action::Emit);
void RenderObjectTree(entt::registry &, entt::entity scene_entity, action::Emit);
void RenderEntityControls(entt::registry &, entt::entity scene_entity, entt::entity active_entity, action::Emit);
