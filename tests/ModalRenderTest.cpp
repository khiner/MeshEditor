// Properties the audio-thread render must hold whatever the contact force model says.
// Each case compares two configurations rendered in one run, or asserts a property of one render.
// No case is written against a stored signal, so changing the model does not invalidate them.

#include "audio/ModalAudio.h"
#include "audio/ModalModes.h"

#include <boost/ut.hpp>

#include <algorithm>
#include <cmath>
#include <span>
#include <vector>

using namespace boost::ut;

namespace {
constexpr float SampleRate{48'000};
constexpr uint32_t BlockSize{512};
constexpr uint32_t SamplePoints{4};

// A contact at rest must reproduce k * delta0^(3/2) == N exactly, with no rounding.
// Powers of two give that: 2^-18 m under 2^31 N/m^(3/2) meets 16 N.
constexpr float RestPenetration{0x1p-18f}, RestStiffness{0x1p31f}, RestLoad{0x1p4f};

// A track with content, so a read costs what it does in the app.
std::shared_ptr<const RoughnessTrack> MakeTrack() {
    RoughnessTrack t;
    t.Heights.resize(TrackSamples);
    t.Sum.resize(TrackSamples + 1, 0.f);
    uint64_t x = 0x9e3779b97f4a7c15ull;
    for (uint32_t i = 0; i < TrackSamples; ++i) {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        t.Heights[i] = float(double(x >> 11) / double(1ull << 52)) - 1.f;
        t.Sum[i + 1] = t.Sum[i] + t.Heights[i];
    }
    t.Spacing = 1e-6f;
    return std::make_shared<const RoughnessTrack>(std::move(t));
}

ModalModes MakeModes(uint32_t mode_count, float longest_t60) {
    ModalModes modes;
    for (uint32_t k = 0; k < mode_count; ++k) {
        modes.Freqs.push_back(40.f * float(k + 1) * 1.031f);
        modes.T60s.push_back(longest_t60 / float(k + 1));
    }
    modes.Shapes.resize(SamplePoints);
    modes.Positions.resize(SamplePoints);
    for (uint32_t p = 0; p < SamplePoints; ++p) {
        modes.Positions[p] = vec3{float(p) * 0.01f, 0.f, 0.f};
        for (uint32_t k = 0; k < mode_count; ++k) {
            const float a = float(k + 1) * 0.37f + float(p);
            modes.Shapes[p].push_back(vec3{std::sin(a), std::cos(a * 1.7f), std::sin(a * 2.3f)} * 0.01f);
        }
    }
    return modes;
}

// A contact riding over both surfaces, as a scrape does.
SustainedState MovingContact(int32_t slot) {
    SustainedState e;
    e.Blend = {.Points = {0, 1, 0}, .Weights = {0.5f, 0.5f, 0.f}};
    e.N = {0.f, 1.f, 0.f};
    e.SlipDir = {1.f, 0.f, 0.f};
    e.SweepDir = {vec3{1.f, 0.f, 0.f}, vec3{0.f, 0.f, -1.f}};
    e.NormalForce = RestLoad;
    e.Friction = 0.5f;
    e.Stiffness = RestStiffness;
    e.StaticPenetration = RestPenetration;
    e.DampingCoeff = 0.4f;
    for (auto &t : e.Tracks) t = {.Index = slot, .Rate = 0.4f, .Sigma = 2e-7f, .Window = 8.f, .Step = 4e-7f};
    return e;
}

// The same contact settled: no surface travel and no slip, so every direction and rate is zero.
SustainedState RestingContact(int32_t slot) {
    auto e = MovingContact(slot);
    e.SlipDir = vec3{0};
    e.SweepDir = {vec3{0}, vec3{0}};
    for (auto &t : e.Tracks) t = {.Index = slot, .Rate = 0.f, .Sigma = 2e-7f, .Window = 8.f, .Step = 0.f};
    return e;
}

// A contact pressing with nothing, which drives no mode however the force model is written.
SustainedState SilentContact() {
    SustainedState e;
    e.Blend = {.Points = {0, 1, 0}, .Weights = {0.5f, 0.5f, 0.f}};
    e.N = {0.f, 1.f, 0.f};
    return e;
}

ModalEvent ImpactEvent(uint32_t object, float impulse) {
    return {.Kind = ModalEventKind::Impact, .Object = object, .ExPos = 0, .Jx = impulse, .Jy = 0.5f * impulse, .Jz = 0.f, .PulseStep = 1.f / 300.f, .PulseGamma = 20.f, .AccelAmp = 0.f};
}

// One bank of identical objects, ready to render.
struct Scene {
    ModalAudio Audio;
    std::vector<uint32_t> Objects;
    int32_t Slot{-1};

    Scene(uint32_t object_count, uint32_t mode_count, float longest_t60, uint32_t renderers) {
        Audio.RenderPool.SetSize(renderers);
        ModalBank next;
        next.SampleRate = SampleRate;
        const auto modes = MakeModes(mode_count, longest_t60);
        for (uint32_t o = 0; o < object_count; ++o) {
            Objects.push_back(AddModalObject(next, entt::entity{o}, modes));
            TuneModalObject(next, Objects.back(), modes.Freqs, modes.T60s);
            next.OutGain[Objects.back()] = 1.f;
        }
        InstallModalBank(Audio, next);
        // Installing a bank flags queued events as addressing the old slot layout, and the next render clears
        // that flag. One silent block does it, before anything is queued against the new bank.
        std::vector<float> discard(BlockSize, 0.f);
        RenderModal(Audio, discard.data(), BlockSize);
        // The install also clears the track pool, so the track is adopted after it.
        BeginSurfaceTrackFrame(Audio);
        Slot = AdoptSurfaceTrack(Audio, 1, [] { return MakeTrack(); });
    }

    // Publish one voice per object, as the frame loop does every step a contact lasts.
    void Publish(const SustainedState &state) {
        auto &set = NextVoiceSet(Audio);
        for (uint32_t o = 0; o < Objects.size(); ++o) set.Voices.emplace_back(uint64_t(o) + 1, Objects[o], state);
        PublishVoiceSet(Audio);
    }

    void Strike(float impulse) {
        for (const auto o : Objects) EnqueueModalEvent(Audio, ImpactEvent(o, impulse));
    }

    // Render `blocks` of `frames` each, republishing `state` before every one, and return the whole signal.
    std::vector<float> Render(uint32_t blocks, uint32_t frames, const SustainedState *state) {
        std::vector<float> signal(size_t(blocks) * frames, 0.f);
        for (uint32_t block = 0; block < blocks; ++block) {
            if (state) Publish(*state);
            RenderModal(Audio, signal.data() + size_t(block) * frames, frames);
        }
        return signal;
    }
};

float Peak(std::span<const float> signal) {
    float peak = 0;
    for (const float s : signal) peak = std::max(peak, std::abs(s));
    return peak;
}

// The largest sample-for-sample gap between two renders.
float MaxDifference(std::span<const float> a, std::span<const float> b) {
    float worst = 0;
    for (size_t i = 0; i < a.size(); ++i) worst = std::max(worst, std::abs(a[i] - b[i]));
    return worst;
}

double Rms(std::span<const float> signal) {
    double energy = 0;
    for (const float s : signal) energy += double(s) * double(s);
    return std::sqrt(energy / double(signal.size()));
}

bool AllFinite(std::span<const float> signal) {
    return std::ranges::all_of(signal, [](float s) { return std::isfinite(s); });
}
} // namespace

int main() {
    // KHR_audio_rigid_bodies Contact Force: with the slip speed and both sweep speeds zero and N constant,
    // the excitation is zero, so a settled body is silent however heavily it is loaded.
    "a contact at rest excites nothing"_test = [] {
        Scene scene{1, 64, 0.2f, 1};
        const auto resting = RestingContact(scene.Slot);
        expect(Peak(scene.Render(8, BlockSize, &resting)) == 0.f);
    };

    // The coupling runs the contact force off the object's own vibration, closing a loop through the mode bank.
    // With no surface travel driving it, that loop is all there is, so a strike's ring must decay away.
    "the coupling loop decays rather than sustaining itself"_test = [] {
        constexpr float LongestT60{0.2f};
        Scene scene{1, 64, LongestT60, 1};
        scene.Strike(1.f);
        const auto resting = RestingContact(scene.Slot);
        const auto blocks = uint32_t(5.f * LongestT60 * SampleRate / BlockSize);
        const auto signal = scene.Render(blocks, BlockSize, &resting);
        const std::span whole{signal};
        expect(AllFinite(whole));
        expect(Peak(whole) > 0.f);
        // Five T60s put a mode's ring-down at 1e-15 of its peak, so anything above that is the loop feeding itself.
        // The contact presses far harder than the strike deflects it, which keeps the loop linear.
        // A degenerate static penetration would not.
        expect(Peak(whole.last(BlockSize)) < Peak(whole) * 1e-9f);
    };

    // A moving contact drives the modes and reads their state back every sample, so a loop gain above unity
    // shows up as a signal that keeps climbing rather than settling.
    "a moving contact settles rather than diverging"_test = [] {
        Scene scene{1, 64, 0.2f, 1};
        const auto moving = MovingContact(scene.Slot);
        const auto signal = scene.Render(256, BlockSize, &moving);
        const std::span whole{signal};
        const auto quarter = whole.size() / 4;
        expect(AllFinite(whole));
        // A second and a half in, a steady contact has settled, so the last quarter cannot outgrow the third.
        expect(Peak(whole.subspan(2 * quarter, quarter)) > 0.f);
        expect(Peak(whole.last(quarter)) < Peak(whole.subspan(2 * quarter, quarter)) * 2.f);
    };

    // The separation is the rigid approach less the object's displacement along the normal, so a rise in force
    // deflects the surface away and takes part of that rise back.
    // A coupled contact is therefore quieter than the same one open loop, and one wired backwards louder.
    // That holds while the contact presses far harder than it deflects. One that lifts clear chatters instead.
    "coupling damps the contact rather than driving it"_test = [] {
        const auto render = [](float coupling) {
            Scene scene{1, 64, 0.2f, 1};
            scene.Audio.Coupling.store(coupling);
            const auto moving = MovingContact(scene.Slot);
            return Rms(scene.Render(256, BlockSize, &moving));
        };
        const auto open_loop = render(0.f), coupled = render(1.f);
        expect(open_loop > 0.);
        expect(coupled < open_loop);
    };

    // A contact pressing with no load and no roughness drives nothing, so a strike must render as if it were absent.
    // It still routes the object through the coupled kernel, which sums modes in a different order from the
    // impact-only one, so the two are close rather than equal.
    "a silent contact leaves a strike alone"_test = [] {
        const auto render = [](bool voice) {
            Scene scene{1, 200, 0.2f, 1};
            scene.Strike(1.f);
            const auto silent = SilentContact();
            return scene.Render(16, BlockSize, voice ? &silent : nullptr);
        };
        const auto without = render(false), with = render(true);
        expect(Peak(without) > 0.f);
        expect(MaxDifference(without, with) < Peak(without) * 1e-5f);
    };

    // The deal is a pure function of the bank, so the renderer count must not change what a block renders.
    // Each renderer sums its own share and the shares are added in renderer order, so the association differs
    // between one renderer and four and the sums are close rather than equal.
    "the render does not depend on how many threads share it"_test = [] {
        const auto render = [](uint32_t renderers) {
            Scene scene{16, 64, 0.2f, renderers};
            scene.Strike(1.f);
            const auto moving = MovingContact(scene.Slot);
            return scene.Render(32, BlockSize, &moving);
        };
        const auto single = render(1), split = render(4);
        expect(Peak(single) > 0.f);
        expect(MaxDifference(single, split) < Peak(single) * 1e-5f);
    };

    // Voice state carries between samples, and a block boundary is the one place it can be dropped.
    // The coupled kernel holds every tuned mode every block, so this is exact.
    // A strike would not be: the impact-only kernel's audible prefix moves with the block length.
    "the render does not depend on where the block boundaries fall"_test = [] {
        const auto render = [](uint32_t blocks, uint32_t frames) {
            Scene scene{1, 64, 0.2f, 1};
            const auto moving = MovingContact(scene.Slot);
            return scene.Render(blocks, frames, &moving);
        };
        const auto whole = render(8, 1024), split = render(32, 256);
        expect(Peak(whole) > 0.f);
        expect(MaxDifference(whole, split) == 0.f);
    };
}
