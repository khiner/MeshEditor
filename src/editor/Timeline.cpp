#include "animation/AnimationTimeline.h"
#include "physics/PhysicsChanges.h"
#include "project/Registry.h"
#include <entt/entity/registry.hpp>
void JumpToStartFrame(entt::registry &r, entt::entity viewport) {
    const auto frame = r.get<const TimelineRange>(viewport).StartFrame;
    project::Patch<TimelinePlayback>(r, viewport, [&](auto &p) { p.CurrentFrame = frame; });
    r.get<PlaybackFrame>(viewport).Value = frame;
    project::EmplaceOrReplace<PhysicsCacheInvalid>(r, viewport);
}
