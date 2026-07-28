#include "audio/ModalAudio.h"

#include <boost/ut.hpp>

#include <algorithm>

using namespace boost::ut;

namespace {
// A track the tests can tell apart by value.
std::shared_ptr<const RoughnessTrack> TrackWith(float spacing) {
    RoughnessTrack track;
    track.Heights.assign(4, 0.f);
    track.Sum.assign(5, 0.f);
    track.Spacing = spacing;
    return std::make_shared<const RoughnessTrack>(std::move(track));
}

size_t FilledSlots(const ModalAudio &m) {
    return size_t(std::ranges::count_if(m.SurfaceTracks, [](const auto &slot) { return bool(slot.Owned); }));
}

// Adopt one distinct track per slot, filling the pool.
void FillPool(ModalAudio &m) {
    for (uint64_t key = 1; key <= ModalAudio::MaxSurfaceTracks; ++key) {
        expect(AdoptSurfaceTrack(m, key, [key] { return TrackWith(float(key)); }) >= 0);
    }
    expect(FilledSlots(m) == size_t(ModalAudio::MaxSurfaceTracks));
}

// A contact event naming one pool slot, so a test can queue work the audio thread has not consumed.
ModalEvent ContactOn(int32_t slot) {
    ModalEvent e{.Kind = ModalEventKind::Contact};
    e.Contact.Tracks[0].Index = slot;
    return e;
}

// Fill the pool, pin one slot the way `pin` says, and check a full frame of fresh adoptions leaves that slot alone.
void ExpectPinnedSlotSurvives(auto &&pin) {
    ModalAudio m;
    BeginSurfaceTrackFrame(m);
    FillPool(m);
    const auto pinned = m.SurfaceTrackSlotByKey.at(1);
    const auto *pinned_track = m.SurfaceTracks[pinned].Owned.get();

    pin(m, pinned);
    BeginSurfaceTrackFrame(m);
    for (uint64_t key = 100; key < 100 + ModalAudio::MaxSurfaceTracks; ++key) {
        AdoptSurfaceTrack(m, key, [key] { return TrackWith(float(key)); });
    }
    expect(m.SurfaceTracks[pinned].Owned.get() == pinned_track);
    expect(m.SurfaceTracks[pinned].Key == 1u);
}
} // namespace

int main() {
    "surfaces with the same finish share one track"_test = [] {
        ModalAudio m;
        BeginSurfaceTrackFrame(m);
        int builds = 0;
        const auto make = [&builds] {
            ++builds;
            return TrackWith(1.f);
        };
        const auto first = AdoptSurfaceTrack(m, 7, make);
        for (int i = 0; i < 8; ++i) expect(AdoptSurfaceTrack(m, 7, make) == first);
        expect(builds == 1);
        expect(FilledSlots(m) == 1u);
    };

    "a track outlives the frame that asked for it"_test = [] {
        ModalAudio m;
        int builds = 0;
        const auto make = [&builds] {
            ++builds;
            return TrackWith(1.f);
        };
        // A contact that stops and resumes finds its track still in the pool rather than rebuilding it.
        BeginSurfaceTrackFrame(m);
        const auto first = AdoptSurfaceTrack(m, 7, make);
        BeginSurfaceTrackFrame(m); // The contact goes quiet for a frame.
        BeginSurfaceTrackFrame(m);
        expect(AdoptSurfaceTrack(m, 7, make) == first);
        expect(builds == 1);
    };

    "a full pool gives up a slot nothing is reading"_test = [] {
        ModalAudio m;
        BeginSurfaceTrackFrame(m);
        FillPool(m);

        // Within the frame that claimed them, every slot is spoken for.
        expect(AdoptSurfaceTrack(m, 999, [] { return TrackWith(9.f); }) == -1);
        expect(m.SurfaceTracksRefused == 1u);

        // Once the frame ends and no voice names them, one gives way.
        BeginSurfaceTrackFrame(m);
        expect(AdoptSurfaceTrack(m, 999, [] { return TrackWith(9.f); }) >= 0);
        expect(FilledSlots(m) == size_t(ModalAudio::MaxSurfaceTracks));
        expect(m.SurfaceTrackSlotByKey.size() == size_t(ModalAudio::MaxSurfaceTracks));
        expect(!m.SurfaceTrackSlotByKey.contains(1)); // The displaced track is gone from the index.
    };

    "a slot a voice is reading is never taken"_test = [] {
        // The audio thread reports a voice reading that slot.
        ExpectPinnedSlotSurvives([](ModalAudio &m, uint32_t slot) { m.VoiceTrackMask.store(1ull << slot); });
    };

    "a slot a queued event names is never taken"_test = [] {
        // The event is enqueued and the audio thread has not consumed it, so no voice names the slot yet.
        ExpectPinnedSlotSurvives([](ModalAudio &m, uint32_t slot) { EnqueueModalEvent(m, ContactOn(int32_t(slot))); });
    };

    "a consumed event stops pinning its slot"_test = [] {
        ModalAudio m;
        BeginSurfaceTrackFrame(m);
        FillPool(m);
        const auto pinned = m.SurfaceTrackSlotByKey.at(1);

        EnqueueModalEvent(m, ContactOn(int32_t(pinned)));
        // The audio thread consumes it and reports that its voice ended.
        m.EventRead.store(m.EventWrite.load());
        m.VoiceTrackMask.store(0);
        BeginSurfaceTrackFrame(m);
        expect(AdoptSurfaceTrack(m, 999, [] { return TrackWith(9.f); }) >= 0);
        expect(!m.SurfaceTrackSlotByKey.contains(1));
    };
}
