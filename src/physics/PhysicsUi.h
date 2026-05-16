#pragma once

#include "Action.h"

#include <functional>

struct PhysicsWorld;

namespace physics_ui {
using ApplyAction = std::function<void(const action::Action &)>;
// Renders the "Physics" tab content (simulation settings + document-level resources).
void RenderTab(entt::registry &, PhysicsWorld &, const ApplyAction &);
// Renders per-entity physics properties (motion type, collider, motion settings).
void RenderEntityProperties(entt::registry &, entt::entity, entt::entity scene_entity, const PhysicsWorld &, const ApplyAction &);
} // namespace physics_ui
