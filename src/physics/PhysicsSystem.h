#pragma once

#include "state/Entity.h"

#include <filesystem>
#include <optional>

struct PhysicsSimulationSettings;
enum class EventPass;

namespace physics {
void Init(state::Scene &);
void Deinit(state::Scene &);
// Rebuild simulation input from changed components.
void ProcessChanges(state::Scene &, EventPass);
// Removes all bodies and constraints while preserving initialization.
void Clear(state::Scene &);

uint32_t BodyCount(const state::Scene &);

// Returns whether `source` membership intersects `target` collision masks.
// Collision requires the test to pass in both directions.
bool DoesFilterAllow(const state::Scene &, state::Entity source, state::Entity target);

void ApplySimulationSettings(state::Scene &, const PhysicsSimulationSettings &);
std::optional<uint32_t> BakedThrough(const state::Scene &);

// Restart the simulation from authored initial conditions on the next playback advance.
void InvalidateCache(state::Scene &);
// Advances playback and returns whether a body pose changed.
bool AdvancePlayback(state::Scene &, state::Entity viewport, int from_frame, int to_frame, int range_start_frame, int range_end_frame, float fps);

// Extends the contiguous cache frontier through `through_frame`, capped at the cache end.
void BakeThrough(state::Scene &, state::Entity viewport, int through_frame, float fps);

// Interpolates cached body poses into WorldTransform at a fractional frame.
void SamplePosesAtFrame(state::Scene &, float frame);
} // namespace physics
