#include "timeline/Timeline.h"
#include "physics/PhysicsSystem.h"
#include "state/Registry.h"
void JumpToStartFrame(state::Registry &r, state::Entity viewport) {
    const auto frame = r.get<const TimelineRange>(viewport).StartFrame;
    r.patch<TimelinePlayback>(viewport, [&](auto &p) { p.CurrentFrame = frame; });
    r.edit<PlaybackFrame>(viewport).Value = frame;
    physics::InvalidateCache(r);
}
