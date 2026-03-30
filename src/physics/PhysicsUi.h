#pragma once

#include "entt_fwd.h"

struct PhysicsWorld;

namespace physics_ui {
// Renders the "Physics" tab content (simulation settings + document-level resources).
void RenderTab(entt::registry &, entt::entity scene_entity, PhysicsWorld &);

// Renders per-entity physics properties (motion type, collider, motion settings).
// `scene_entity` is the singleton entity holding AnimationTimeline.
void RenderEntityProperties(entt::registry &, entt::entity entity, entt::entity scene_entity, PhysicsWorld &);
} // namespace physics_ui
