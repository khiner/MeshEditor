#include "animation/AnimationTimeline.h"
#include "physics/PhysicsSystem.h"
#include "state/Scene.h"
void JumpToStartFrame(state::Scene &r, state::Entity viewport) {
    const auto frame = r.get<const TimelineRange>(viewport).StartFrame;
    r.patch<TimelinePlayback>(viewport, [&](auto &p) { p.CurrentFrame = frame; });
    r.edit<PlaybackFrame>(viewport).Value = frame;
    physics::InvalidateCache(r);
}
