#pragma once

#include "Action.h" // action::Emit

struct PhysicsWorld;

namespace physics_ui {
void RenderTab(entt::registry &, entt::entity viewport, PhysicsWorld &, action::Emit);
void RenderEntityProperties(entt::registry &, entt::entity, entt::entity viewport, const PhysicsWorld &, action::Emit);
} // namespace physics_ui
