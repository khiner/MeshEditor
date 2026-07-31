// Times the audio-thread render kernels against the block deadline they have to meet.
// The coupled kernel reads each voice's deflection back out of the mode bank every sample, so it cannot hoist its
// excitation gains the way the impact-only kernel does. That difference is what this measures.

#include "audio/ModalAudio.h"
#include "audio/ModalModes.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
constexpr float SampleRate{48'000};
constexpr uint32_t BlockSize{512};
constexpr uint32_t SamplePoints{4};

// A positive count named by the environment, held at `floor`, or `fallback` when it names none.
uint32_t EnvCount(const char *name, uint32_t fallback, uint32_t floor = 1) {
    const char *env = std::getenv(name);
    const auto n = env ? std::atoi(env) : 0;
    return n > 0 ? std::max(floor, uint32_t(n)) : fallback;
}

// How many renderers a block is split between, so one sweep can be diffed against another.
const uint32_t RenderCount = EnvCount("MODAL_RENDERERS", 1);

// Blocks each case renders. The default is what a timing run needs to settle, and a checksum comparison can ask for far fewer.
// The floor of 128 is what `struck every 32` needs to strike more than once and what the ring-down cases need
// for the audibility gate to engage.
const uint32_t Blocks = EnvCount("MODAL_BLOCKS", 2000, 128);

// A track with content, so the read path costs what it costs in the app.
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

// `live` modes sound and the rest are pushed past Nyquist, as in a bank baked at one scale and retuned upward.
ModalModes MakeModes(uint32_t mode_count, uint32_t live) {
    ModalModes modes;
    modes.Freqs.reserve(mode_count);
    modes.T60s.reserve(mode_count);
    for (uint32_t k = 0; k < mode_count; ++k) {
        modes.Freqs.push_back(k < live ? 40.f * float(k + 1) * 1.031f : SampleRate);
        modes.T60s.push_back(1.5f / float(k + 1));
    }
    modes.Shapes.resize(SamplePoints);
    modes.Positions.resize(SamplePoints);
    for (uint32_t p = 0; p < SamplePoints; ++p) {
        modes.Positions[p] = vec3{float(p) * 0.01f, 0.f, 0.f};
        modes.Shapes[p].reserve(mode_count);
        for (uint32_t k = 0; k < mode_count; ++k) {
            const float a = float(k + 1) * 0.37f + float(p);
            modes.Shapes[p].push_back(vec3{std::sin(a), std::cos(a * 1.7f), std::sin(a * 2.3f)} * 0.01f);
        }
    }
    return modes;
}

// One contact on the object, refreshed the way the frame loop refreshes it every step the contact lasts.
// A slot of -1 leaves the voice with no surface to ride, which costs the force model without costing a track read.
SustainedState ContactState(int32_t slot) {
    SustainedState e;
    e.Blend = {.Points = {0, 1, 0}, .Weights = {0.5f, 0.5f, 0.f}};
    e.N = {0.f, 1.f, 0.f};
    e.SlipDir = {1.f, 0.f, 0.f};
    // The two surfaces sweep along different directions, so each geometric term takes its own drive row.
    e.SweepDir = {vec3{1.f, 0.f, 0.f}, vec3{0.f, 0.f, 1.f}};
    e.NormalForce = 5.f;
    e.Friction = 0.5f;
    e.Stiffness = 2e9f;
    e.StaticPenetration = 2e-6f;
    e.DampingCoeff = 0.4f;
    // A patch wider than the track spacing, which is the windowed read the app takes.
    for (auto &t : e.Tracks) t = {.Index = slot, .Rate = 0.4f, .Sigma = 2e-7f, .Window = 8.f, .Step = 4e-7f};
    return e;
}

ModalEvent ImpactEvent(uint32_t object) {
    return {.Kind = ModalEventKind::Impact, .Object = object, .ExPos = 0, .Jx = 1.f, .Jy = 0.5f, .Jz = 0.f, .PulseStep = 1.f / 300.f, .PulseGamma = 20.f, .AccelAmp = 0.f};
}

struct Result {
    double MicrosPerBlock;
    double Checksum;
};

struct Case {
    const char *Label;
    uint32_t Modes, Live, Objects, Excited, Voices, Impacts;
    bool Tracks;
    // Blocks between excitations. One re-excites every block, which holds every object at full amplitude.
    // Higher leaves the objects ringing freely in between, which is where a mode decays out of earshot.
    uint32_t StrikeEvery{1};
};

// `Voices` contacts and `Impacts` in-flight strikes on each of `Objects` objects, timed over `blocks` render calls.
Result Bench(const Case &c, uint32_t blocks) {
    const auto [label, mode_count, live, object_count, excited, voices, impacts, tracks, strike_every] = c;
    ModalAudio m;
    // The impact cap is a scene setting rather than a kernel cost, so it is lifted clear of every case here.
    // The voice cap applies where the set is built, which this bench writes past.
    m.MaxImpacts.store(1u << 20);
    m.RenderPool.SetSize(RenderCount);
    ModalBank next;
    next.SampleRate = SampleRate;
    const auto modes = MakeModes(mode_count, live);
    std::vector<uint32_t> objects;
    for (uint32_t o = 0; o < object_count; ++o) {
        objects.push_back(AddModalObject(next, entt::entity{o}, modes));
        TuneModalObject(next, objects.back(), modes.Freqs, modes.T60s);
        next.OutGain[objects.back()] = 1.f;
    }
    InstallModalBank(m, next);

    // The pool is cleared by the install above, so the track is adopted after it.
    BeginSurfaceTrackFrame(m);
    const auto slot = tracks ? AdoptSurfaceTrack(m, 1, [] { return MakeTrack(); }) : -1;

    std::vector<float> out(BlockSize);
    const auto refresh = [&] {
        auto &set = NextVoiceSet(m);
        for (uint32_t o = 0; o < excited; ++o) {
            for (uint32_t v = 0; v < voices; ++v) {
                auto &voice = set.Voices.emplace_back(uint64_t(o) * voices + v + 1, objects[o], ContactState(slot));
                voice.State.Blend.Points[0] = v % SamplePoints;
            }
            for (uint32_t i = 0; i < impacts; ++i) EnqueueModalEvent(m, ImpactEvent(objects[o]));
        }
        PublishVoiceSet(m);
    };

    for (uint32_t warm = 0; warm < 16; ++warm) {
        refresh();
        std::fill(out.begin(), out.end(), 0.f);
        RenderModal(m, out.data(), BlockSize);
    }

    double checksum = 0;
    const auto start = std::chrono::steady_clock::now();
    for (uint32_t block = 0; block < blocks; ++block) {
        if (block % strike_every == 0) refresh();
        std::fill(out.begin(), out.end(), 0.f);
        RenderModal(m, out.data(), BlockSize);
        for (const float s : out) checksum += double(s);
    }
    const auto elapsed = std::chrono::duration<double, std::micro>{std::chrono::steady_clock::now() - start}.count();
    return {elapsed / double(blocks), checksum};
}
} // namespace

int main() {
    constexpr Case Cases[]{
        {"64 modes, 1 voice", 64, 64, 1, 1, 1, 0, true},
        {"64 modes, 4 voices", 64, 64, 1, 1, 4, 0, true},
        {"200 modes, 1 voice", 200, 200, 1, 1, 1, 0, true},
        {"200 modes, 1 voice, 1 impact", 200, 200, 1, 1, 1, 1, true},
        {"200 modes, 2 voices", 200, 200, 1, 1, 2, 0, true},
        {"200 modes, 4 voices", 200, 200, 1, 1, 4, 0, true},
        {"200 modes, 1 voice, no tracks", 200, 200, 1, 1, 1, 0, false},
        {"512 modes, 1 voice", 512, 512, 1, 1, 1, 0, true},
        {"512 modes, 4 voices", 512, 512, 1, 1, 4, 0, true},
        {"512 modes 200 live, 1 voice", 512, 200, 1, 1, 1, 0, true},
        {"64 modes, impact only", 64, 64, 1, 1, 0, 1, true},
        {"200 modes, impact only", 200, 200, 1, 1, 0, 1, true},
        {"200 modes, 4 impacts", 200, 200, 1, 1, 0, 4, true},
        {"512 modes, impact only", 512, 512, 1, 1, 0, 1, true},
        // Many voices on one object, which is the coupled kernel's own scaling.
        {"200 modes, 16 voices", 200, 200, 1, 1, 16, 0, true},
        {"200 modes, 64 voices", 200, 200, 1, 1, 64, 0, true},
        {"200 modes, 256 voices", 200, 200, 1, 1, 256, 0, true},
        // The same voice counts spread one per object, which is the shape a scene actually has.
        {"200 modes, 16 objects, 1 voice", 200, 200, 16, 16, 1, 0, true},
        {"200 modes, 64 objects, 1 voice", 200, 200, 64, 64, 1, 0, true},
        {"200 modes, 256 objects, 1 voice", 200, 200, 256, 256, 1, 0, true},
        // The same objects ringing with no voice at all, which separates the mode advance from the contact.
        {"200 modes, 64 objects, impact only", 200, 200, 64, 64, 0, 1, true},
        {"200 modes, 256 objects, impact only", 200, 200, 256, 256, 0, 1, true},
        // A bank of objects where only a few still sound, as a scene looks a moment after anything happens.
        {"200 modes, 256 objects, 16 ringing", 200, 200, 256, 16, 16, 0, true},
        {"200 modes, 1024 objects, 16 ringing", 200, 200, 1024, 16, 16, 0, true},
        // Struck at intervals and left to ring in between, as an impact-driven scene does.
        // T60 runs 1.5 s at the fundamental down to 7.5 ms at mode 200, so most of a ring-down is a few low modes.
        {"200 modes, 64 objects, struck every 32", 200, 200, 64, 64, 0, 1, true, 32},
        {"200 modes, 256 objects, struck every 32", 200, 200, 256, 256, 0, 1, true, 32},
    };
    // Real time one block of audio covers, which is the deadline every case is measured against.
    const double block_micros = 1e6 * double(BlockSize) / double(SampleRate);
    std::printf("%-32s %10s %9s %16s\n", "case", "us/block", "realtime", "checksum");
    for (const auto &c : Cases) {
        const auto r = Bench(c, Blocks);
        std::printf("%-32s %10.2f %8.0fx %16.6f\n", c.Label, r.MicrosPerBlock, block_micros / r.MicrosPerBlock, r.Checksum);
    }
    return 0;
}
