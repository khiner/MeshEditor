#pragma once

#include <entt/entity/fwd.hpp>

struct FrameState;

// Mouse/keyboard interaction for the viewport. Mutates the per-frame Frame state.
void Interact(entt::registry &, entt::entity viewport, FrameState &Frame);
// Overlay controls (gizmos, mode pickers). Mutates Frame.
void InteractOverlay(entt::registry &, entt::entity viewport, FrameState &Frame);
// Reads Frame state set by InteractOverlay and renders the overlay (gizmo, box-select rectangle, origin dots).
void DrawOverlay(entt::registry &, entt::entity viewport, FrameState &Frame);
// Side-panel controls (interaction mode, object tree, materials, etc.).
void RenderControls(entt::registry &, entt::entity viewport);

void RenderClipPickers(entt::registry &);
void RenderObjectTree(entt::registry &, entt::entity viewport);
void RenderEntityControls(entt::registry &, entt::entity viewport, entt::entity active_entity);
