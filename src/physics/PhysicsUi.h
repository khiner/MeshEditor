#pragma once

#include "Action.h" // action::Emit

struct PhysicsWorld;

namespace physics_ui {
// Renders the "Physics" tab content (simulation settings + document-level resources).
void RenderTab(entt::registry &, entt::entity viewport, PhysicsWorld &, action::Emit);
// Renders per-entity physics properties (motion type, collider, motion settings).
void RenderEntityProperties(entt::registry &, entt::entity, entt::entity viewport, const PhysicsWorld &, action::Emit);
} // namespace physics_ui
