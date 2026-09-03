#pragma once

#include <entt/entity/fwd.hpp>

namespace action {
// Applies the buffered user transition followed by system-generated actions.
void ApplyEmitted(entt::registry &, entt::entity viewport);

// Applies and records immediately outside frame emission.
template<typename ActionType> void ApplyNow(entt::registry &, entt::entity viewport, ActionType);
} // namespace action
