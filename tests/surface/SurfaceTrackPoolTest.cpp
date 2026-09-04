#include "audio/surface/SurfaceModel.h"

#include "RunSuites.h"

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
    return size_t(std::ranges::count_if(Surface(m).SurfaceTracks, [](const auto &slot) { return bool(slot.Owned); }));
}

// Adopt one distinct track per slot, filling the pool.
void FillPool(ModalAudio &m) {
    for (uint64_t key = 1; key <= SurfaceAudioState::MaxSurfaceTracks; ++key) {
        expect(AdoptSurfaceTrack(Surface(m), key, [key] { return TrackWith(float(key)); }) >= 0);
    }
    expect(FilledSlots(m) == size_t(SurfaceAudioState::MaxSurfaceTracks));
}

void PublishContactOn(ModalAudio &m, int32_t slot) {
    auto &set = NextVoiceSet(Surface(m));
    set.Voices.emplace_back(1, 0).State.Tracks[0].Index = slot;
    PublishVoiceSet(Surface(m));
}

void PublishNothing(ModalAudio &m) {
    for (size_t i = 0; i < Surface(m).VoiceSets.size(); ++i) {
        NextVoiceSet(Surface(m));
        PublishVoiceSet(Surface(m));
    }
}

// Fill the pool, pin one slot as `pin` specifies, and check a full frame of fresh adoptions leaves that slot alone.
void ExpectPinnedSlotSurvives(auto &&pin) {
    ModalAudio m;
    BeginSurfaceTrackFrame(Surface(m));
    FillPool(m);
    const auto pinned = Surface(m).SurfaceTrackSlotByKey.at(1);
    const auto *pinned_track = Surface(m).SurfaceTracks[pinned].Owned.get();

    pin(m, pinned);
    BeginSurfaceTrackFrame(Surface(m));
    for (uint64_t key = 100; key < 100 + SurfaceAudioState::MaxSurfaceTracks; ++key) {
        AdoptSurfaceTrack(Surface(m), key, [key] { return TrackWith(float(key)); });
    }
    expect(Surface(m).SurfaceTracks[pinned].Owned.get() == pinned_track);
    expect(Surface(m).SurfaceTracks[pinned].Key == 1u);
}
} // namespace
int main() {
    "surfaces with the same finish share one track"_test = [] {
        ModalAudio m;
        BeginSurfaceTrackFrame(Surface(m));
        int builds = 0;
        const auto make = [&builds] {
            ++builds;
            return TrackWith(1.f);
        };
        const auto first = AdoptSurfaceTrack(Surface(m), 7, make);
        for (int i = 0; i < 8; ++i) expect(AdoptSurfaceTrack(Surface(m), 7, make) == first);
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
        BeginSurfaceTrackFrame(Surface(m));
        const auto first = AdoptSurfaceTrack(Surface(m), 7, make);
        BeginSurfaceTrackFrame(Surface(m)); // The contact goes quiet for a frame.
        BeginSurfaceTrackFrame(Surface(m));
        expect(AdoptSurfaceTrack(Surface(m), 7, make) == first);
        expect(builds == 1);
    };

    "a full pool gives up a slot nothing is reading"_test = [] {
        ModalAudio m;
        BeginSurfaceTrackFrame(Surface(m));
        FillPool(m);

        // Within the frame that claimed them, every slot is taken.
        expect(AdoptSurfaceTrack(Surface(m), 999, [] { return TrackWith(9.f); }) == -1);
        expect(Surface(m).SurfaceTracksRefused == 1u);

        // Once the frame ends and no voice names them, one is reused.
        BeginSurfaceTrackFrame(Surface(m));
        expect(AdoptSurfaceTrack(Surface(m), 999, [] { return TrackWith(9.f); }) >= 0);
        expect(FilledSlots(m) == size_t(SurfaceAudioState::MaxSurfaceTracks));
        expect(Surface(m).SurfaceTrackSlotByKey.size() == size_t(SurfaceAudioState::MaxSurfaceTracks));
        expect(!Surface(m).SurfaceTrackSlotByKey.contains(1)); // The displaced track is gone from the index.
    };

    "a slot a voice is reading is never taken"_test = [] {
        // The audio thread reports a voice reading that slot.
        ExpectPinnedSlotSurvives([](ModalAudio &m, uint32_t slot) { Surface(m).VoiceTrackMask.store(1ull << slot); });
    };

    "a slot a published contact names is never taken"_test = [] {
        // The set is published and the audio thread has not adopted it, so no voice names the slot yet.
        ExpectPinnedSlotSurvives([](ModalAudio &m, uint32_t slot) { PublishContactOn(m, int32_t(slot)); });
    };

    "a contact that stops being published stops pinning its slot"_test = [] {
        ModalAudio m;
        BeginSurfaceTrackFrame(Surface(m));
        FillPool(m);
        const auto pinned = Surface(m).SurfaceTrackSlotByKey.at(1);

        PublishContactOn(m, int32_t(pinned));
        // The audio thread adopts it and reports that its voice ended, and the set itself rotates out.
        Surface(m).VoiceTrackMask.store(0);
        PublishNothing(m);
        BeginSurfaceTrackFrame(Surface(m));
        expect(AdoptSurfaceTrack(Surface(m), 999, [] { return TrackWith(9.f); }) >= 0);
        expect(!Surface(m).SurfaceTrackSlotByKey.contains(1));
    };

    return RunSuites();
}
