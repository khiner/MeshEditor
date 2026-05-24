#pragma once

#include <entt/entity/fwd.hpp>

struct PhysicsWorld;

namespace physics_ui {
void RenderTab(entt::registry &, entt::entity viewport, PhysicsWorld &);
void RenderEntityProperties(entt::registry &, entt::entity, entt::entity viewport, const PhysicsWorld &);
} // namespace physics_ui
