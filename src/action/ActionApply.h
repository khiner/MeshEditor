#pragma once

#include <entt/entity/fwd.hpp>

namespace action {
// Apply this frame's buffered action / gesture transition (if any).
void ApplyEmitted(entt::registry &, entt::entity viewport);

// Synchronously apply + record a playback-stop if playing.
void StopPlaybackIfPlaying(entt::registry &, entt::entity viewport);
} // namespace action
