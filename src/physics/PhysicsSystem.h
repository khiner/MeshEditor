#pragma once

#include "state/Entity.h"

#include <filesystem>
#include <optional>

struct PhysicsSimulationSettings;

// The references to each physics material, collision system, collision filter, and joint definition.
struct PhysicsDefinitionUses {
    std::unordered_map<state::Entity, uint32_t> Counts;

    uint32_t Count(state::Entity definition) const {
        const auto it = Counts.find(definition);
        return it != Counts.end() ? it->second : 0u;
    }
};

namespace physics {
void Init(state::Scene &);
void Deinit(state::Scene &);
// Updates the bodies and joints that changed components reach, and the definition use counts.
// A body whose inputs changed recooks, updates in place or takes new surfaces, and any change invalidates the cache.
void ProcessChanges(state::Scene &);
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
// An invalid cache restarts at once and bakes on the next frame change, showing the cached start until then.
bool AdvancePlayback(state::Scene &, state::Entity viewport, int from_frame, int to_frame, int range_start_frame, int range_end_frame, float fps);

// Extends the contiguous cache frontier through `through_frame`, capped at the cache end.
void BakeThrough(state::Scene &, state::Entity viewport, int through_frame, float fps);

// Interpolates cached body poses into the bodies' world transforms at a fractional frame.
void SamplePosesAtFrame(state::Scene &, float frame);
} // namespace physics
