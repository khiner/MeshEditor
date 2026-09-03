#pragma once

#include <entt/entity/fwd.hpp>

struct FrameState;

void Interact(entt::registry &, entt::entity viewport, FrameState &Frame);
void InteractOverlay(entt::registry &, entt::entity viewport, FrameState &Frame);
void DrawOverlay(entt::registry &, entt::entity viewport, FrameState &Frame);
