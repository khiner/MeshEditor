#pragma once

#include "action/Physics.h"

#include <optional>

struct PhysicsWorld;

namespace physics_ui {
// Renders the "Physics" tab content (simulation settings + document-level resources).
// Returns an action for the caller to apply, if the user triggered one this frame.
std::optional<action::physics::Action> RenderTab(entt::registry &, entt::entity scene_entity, PhysicsWorld &);
// Renders per-entity physics properties (motion type, collider, motion settings).
// Returns an action for the caller to apply, if the user triggered one this frame.
std::optional<action::physics::Action> RenderEntityProperties(entt::registry &, entt::entity, entt::entity scene_entity, const PhysicsWorld &);
} // namespace physics_ui
