#pragma once

#include <entt/entity/fwd.hpp>

namespace action {
// Apply this frame's buffered user action / gesture transition (if any), then its system-generated actions.
void ApplyEmitted(entt::registry &, entt::entity viewport);

// Apply and record an action immediately, outside the per-frame emission.
// For orchestration points outside the frame's UI pass (scene seeding, quiesce). UI code emits instead.
template<typename ActionType> void ApplyNow(entt::registry &, entt::entity viewport, ActionType);
} // namespace action
