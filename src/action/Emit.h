#pragma once

#include <entt/entity/fwd.hpp>

#include <cstddef>

namespace action {
// Stage an action to execute at the end of the frame.
// First action emitted per-frame is executed and the rest are ignored.
template<typename ActionType> void Emit(ActionType);

// Apply the first action emitted this frame (if any).
void ApplyEmitted(entt::registry &, entt::entity viewport);

std::size_t ActionSize();
} // namespace action
