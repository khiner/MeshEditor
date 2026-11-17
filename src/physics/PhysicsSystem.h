#pragma once

#include "physics/PhysicsSchema.h"
#include "state/Entity.h"

struct PhysicsSimulationSettings;

namespace physics {
void Init(state::Registry &);
void Deinit(state::Registry &);
// Updates the bodies and joints that changed components reach, and the definition use counts.
// A body whose inputs changed recooks, updates in place or takes new surfaces, and any change invalidates the cache.
void ProcessChanges(state::Registry &);
uint32_t BodyCount(const state::Registry &);

// Returns whether `source` membership intersects `target` collision masks.
// Collision requires the test to pass in both directions.
bool DoesFilterAllow(const state::Registry &, state::Entity source, state::Entity target);

void ApplySimulationSettings(state::Registry &, const PhysicsSimulationSettings &);
std::optional<uint32_t> BakedThrough(const state::Registry &);

// Restart the simulation from authored initial conditions on the next playback advance.
void InvalidateCache(state::Registry &);
// Advances playback and returns whether a body pose changed.
// An invalid cache restarts at once and bakes on the next frame change, showing the cached start until then.
bool AdvancePlayback(state::Registry &, int from_frame, int to_frame, int range_start_frame, int range_end_frame, float fps);

// Extends the contiguous cache frontier through `through_frame`, capped at the cache end.
void BakeThrough(state::Registry &, int through_frame, float fps);

// Interpolates cached body poses into the bodies' world transforms at a fractional frame.
void SamplePosesAtFrame(state::Registry &, float frame);
} // namespace physics
