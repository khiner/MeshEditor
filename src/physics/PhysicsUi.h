#pragma once

#include "action/Physics.h"

#include <functional>

struct PhysicsWorld;

namespace physics_ui {
// By value (not const&) so move-only alternatives (e.g. Replace<PhysicsMotion>) survive the visit-into-Apply hop.
using ApplyAction = std::function<void(action::physics::Action)>;
// Renders the "Physics" tab content (simulation settings + document-level resources).
void RenderTab(entt::registry &, entt::entity scene_entity, PhysicsWorld &, const ApplyAction &);
// Renders per-entity physics properties (motion type, collider, motion settings).
void RenderEntityProperties(entt::registry &, entt::entity, entt::entity scene_entity, const PhysicsWorld &, const ApplyAction &);
} // namespace physics_ui
