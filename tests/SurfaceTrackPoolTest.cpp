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

// Publish one contact naming one pool slot, so a test can hold a slot the audio thread has not adopted yet.
void PublishContactOn(ModalAudio &m, int32_t slot) {
    auto &set = NextVoiceSet(m);
    set.Voices.emplace_back(1, 0).State.Tracks[0].Index = slot;
    PublishVoiceSet(m);
}

// Publish nothing until every set the audio thread could hold has rotated out.
void PublishNothing(ModalAudio &m) {
    for (size_t i = 0; i < m.VoiceSets.size(); ++i) {
        NextVoiceSet(m);
        PublishVoiceSet(m);
    }
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

    "a slot a published contact names is never taken"_test = [] {
        // The set is published and the audio thread has not adopted it, so no voice names the slot yet.
        ExpectPinnedSlotSurvives([](ModalAudio &m, uint32_t slot) { PublishContactOn(m, int32_t(slot)); });
    };

    "a contact that stops being published stops pinning its slot"_test = [] {
        ModalAudio m;
        BeginSurfaceTrackFrame(m);
        FillPool(m);
        const auto pinned = m.SurfaceTrackSlotByKey.at(1);

        PublishContactOn(m, int32_t(pinned));
        // The audio thread adopts it and reports that its voice ended, and the set itself rotates out.
        m.VoiceTrackMask.store(0);
        PublishNothing(m);
        BeginSurfaceTrackFrame(m);
        expect(AdoptSurfaceTrack(m, 999, [] { return TrackWith(9.f); }) >= 0);
        expect(!m.SurfaceTrackSlotByKey.contains(1));
    };
}
