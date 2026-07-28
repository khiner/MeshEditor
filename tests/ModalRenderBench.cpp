// Times the audio-thread render kernels against the block deadline they have to meet.
// The coupled kernel reads each voice's deflection back out of the mode bank every sample, so it cannot hoist its
// excitation gains the way the impact-only kernel does, which is what this measures.

#include "audio/ModalAudio.h"
#include "audio/ModalModes.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {
constexpr float SampleRate{48'000};
constexpr uint32_t BlockSize{512};
constexpr uint32_t SamplePoints{4};

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

// `live` modes sound and the rest are pushed past Nyquist, which is what a bank baked at one scale and retuned upward looks like.
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
ModalEvent ContactEvent(uint32_t object, int32_t slot) {
    ModalEvent e{.Kind = ModalEventKind::Contact, .Object = object, .ContactId = 1};
    e.Contact.Blend = {.First = 0, .Second = 1, .Weight = 0.5f};
    e.Contact.N = {0.f, 1.f, 0.f};
    e.Contact.Slip = {0.3f, 0.f, 0.f};
    e.Contact.NormalForce = 5.f;
    e.Contact.Stiffness = 2e9f;
    e.Contact.StaticPenetration = 2e-6f;
    e.Contact.DampingCoeff = 0.4f;
    // A patch wider than the track spacing, which is the windowed read the app takes.
    for (auto &t : e.Contact.Tracks) t = {.Index = slot, .Rate = 0.4f, .Sigma = 2e-7f, .Window = 8.f, .Step = 4e-7f};
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
    uint32_t Modes, Live, Voices, Impacts;
    bool Tracks;
};

// `Voices` contacts and `Impacts` in-flight strikes on one object, timed over `blocks` render calls.
Result Bench(const Case &c, uint32_t blocks) {
    const auto [label, mode_count, live, voices, impacts, tracks] = c;
    ModalAudio m;
    ModalBank next;
    next.SampleRate = SampleRate;
    const auto modes = MakeModes(mode_count, live);
    const auto object = AddModalObject(next, entt::entity{0}, modes);
    TuneModalObject(next, object, modes.Freqs, modes.T60s);
    next.OutGain[object] = 1.f;
    InstallModalBank(m, next);

    // The pool is cleared by the install above, so the track is adopted after it.
    BeginSurfaceTrackFrame(m);
    const auto slot = tracks ? AdoptSurfaceTrack(m, 1, [] { return MakeTrack(); }) : -1;

    std::vector<float> out(BlockSize);
    const auto refresh = [&] {
        for (uint32_t v = 0; v < voices; ++v) {
            auto e = ContactEvent(object, slot);
            e.ContactId = v + 1;
            e.Contact.Blend.First = v % SamplePoints;
            EnqueueModalEvent(m, e);
        }
        for (uint32_t i = 0; i < impacts; ++i) EnqueueModalEvent(m, ImpactEvent(object));
    };

    for (uint32_t warm = 0; warm < 16; ++warm) {
        refresh();
        std::fill(out.begin(), out.end(), 0.f);
        RenderModal(m, out.data(), BlockSize);
    }

    double checksum = 0;
    const auto start = std::chrono::steady_clock::now();
    for (uint32_t block = 0; block < blocks; ++block) {
        refresh();
        std::fill(out.begin(), out.end(), 0.f);
        RenderModal(m, out.data(), BlockSize);
        for (const float s : out) checksum += double(s);
    }
    const auto elapsed = std::chrono::duration<double, std::micro>{std::chrono::steady_clock::now() - start}.count();
    return {elapsed / double(blocks), checksum};
}
} // namespace

int main() {
    constexpr uint32_t Blocks{2000};
    constexpr Case Cases[]{
        {"64 modes, 1 voice", 64, 64, 1, 0, true},
        {"64 modes, 4 voices", 64, 64, 4, 0, true},
        {"200 modes, 1 voice", 200, 200, 1, 0, true},
        {"200 modes, 1 voice, 1 impact", 200, 200, 1, 1, true},
        {"200 modes, 2 voices", 200, 200, 2, 0, true},
        {"200 modes, 4 voices", 200, 200, 4, 0, true},
        {"200 modes, 1 voice, no tracks", 200, 200, 1, 0, false},
        {"512 modes, 1 voice", 512, 512, 1, 0, true},
        {"512 modes, 4 voices", 512, 512, 4, 0, true},
        {"512 modes 200 live, 1 voice", 512, 200, 1, 0, true},
        {"64 modes, impact only", 64, 64, 0, 1, true},
        {"200 modes, impact only", 200, 200, 0, 1, true},
        {"200 modes, 4 impacts", 200, 200, 0, 4, true},
        {"512 modes, impact only", 512, 512, 0, 1, true},
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
