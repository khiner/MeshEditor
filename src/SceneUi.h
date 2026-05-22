#pragma once

#include "Action.h"

#include <entt/entity/fwd.hpp>

void RenderClipPickers(entt::registry &, action::Emit);
void RenderObjectTree(entt::registry &, entt::entity scene_entity, action::Emit);
void RenderEntityControls(entt::registry &, entt::entity scene_entity, entt::entity active_entity, action::Emit);
