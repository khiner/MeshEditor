#include "timeline/Timeline.h"
#include "physics/PhysicsSystem.h"
#include "state/Registry.h"
void JumpToStartFrame(state::Registry &r) {
    const auto frame = r.get<const TimelineRange>(state::Document).StartFrame;
    r.patch<TimelinePlayback>(state::Document, [&](auto &p) { p.CurrentFrame = frame; });
    r.edit<PlaybackFrame>(state::Document).Value = frame;
    physics::InvalidateCache(r);
}
