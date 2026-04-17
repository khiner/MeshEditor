#pragma once

#include "entt_fwd.h"

struct PhysicsWorld;

namespace physics_ui {
// Renders the "Physics" tab content (simulation settings + document-level resources).
void RenderTab(entt::registry &, PhysicsWorld &);
// Renders per-entity physics properties (motion type, collider, motion settings).
void RenderEntityProperties(entt::registry &, entt::entity, entt::entity scene_entity);
} // namespace physics_ui
