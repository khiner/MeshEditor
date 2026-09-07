#include "animation/AnimationTimeline.h"
#include "physics/PhysicsChanges.h"
#include <entt/entity/registry.hpp>
void JumpToStartFrame(entt::registry &r, entt::entity viewport) {
    const auto frame = r.get<const TimelineRange>(viewport).StartFrame;
    r.patch<TimelinePlayback>(viewport, [&](auto &p) { p.CurrentFrame = frame; });
    r.get<PlaybackFrame>(viewport).Value = frame;
    r.emplace_or_replace<PhysicsCacheInvalid>(viewport);
}
