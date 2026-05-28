#pragma once

#include <entt/entity/fwd.hpp>

#include <cstdint>
#include <optional>

struct PhysicsSimulationSettings;

namespace physics {
void Init(entt::registry &);
void Deinit(entt::registry &);

uint32_t BodyCount(const entt::registry &);

// Directional single-side test: does `source`'s membership intersect `target`'s collide-mask?
// Effective collision requires this in both directions. Use for UI asymmetry visualization.
bool DoesFilterAllow(const entt::registry &, entt::entity source, entt::entity target);

void ApplySimulationSettings(entt::registry &, const PhysicsSimulationSettings &);
std::optional<uint32_t> BakedThrough(const entt::registry &); // nullopt if nothing baked since last clear

// Advance physics playback from from_frame to to_frame. Returns true if a body pose changed.
bool AdvancePlayback(entt::registry &, entt::entity viewport, int from_frame, int to_frame, int range_start_frame, int range_end_frame, float fps, bool range_changed, bool cache_invalid);
} // namespace physics
