#pragma once

#include <entt/entity/fwd.hpp>

namespace physics_ui {
void RenderTab(entt::registry &, entt::entity viewport);
void RenderEntityProperties(entt::registry &, entt::entity, entt::entity viewport);
} // namespace physics_ui
