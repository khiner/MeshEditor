#pragma once

#include "audio/surface/SurfaceModel.h"

#include "audio/ContactSurface.h"

#include "ModalBench.h"

#include <cmath>
#include <memory>
#include <span>
#include <vector>

inline std::shared_ptr<const RoughnessTrack> MakeTrack() {
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

// A modal bank carrying a track pool, ready to publish voices into.
struct Scene : ModalScene {
    using ModalScene::Render;

    int32_t Slot{-1};

    Scene(uint32_t object_count, uint32_t mode_count, float longest_t60, uint32_t renderers, float rigid_inv_mass = 0.f, float sample_rate = SampleRate, float shape_scale = 1.f)
        : Scene(MakeModes(mode_count, longest_t60, shape_scale), object_count, renderers, rigid_inv_mass, sample_rate) {}

    // One bank of whatever modes a case needs to press, for a bench that needs its own frequency.
    Scene(const ModalModes &modes, uint32_t object_count, uint32_t renderers, float rigid_inv_mass, float sample_rate = SampleRate)
        : ModalScene(modes, object_count, renderers, rigid_inv_mass, sample_rate) {
        // The install also clears the track pool, so the track is adopted after it.
        BeginSurfaceTrackFrame(Surface(Audio));
        Slot = AdoptSurfaceTrack(Surface(Audio), 1, [] { return MakeTrack(); });
    }

    // Publish one voice per object, as the frame loop does every step a contact lasts.
    void Publish(const SustainedState &state) {
        auto &set = NextVoiceSet(Surface(Audio));
        for (uint32_t o = 0; o < Objects.size(); ++o) set.Voices.emplace_back(uint64_t(o) + 1, Objects[o], state);
        PublishVoiceSet(Surface(Audio));
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

// The seated contact's axes, which the states below hand the voice as its normal and its resting slip direction.
constexpr vec3 ContactNormal{0.f, 1.f, 0.f}, ContactTangent{1.f, 0.f, 0.f};

constexpr float SeatedModeShape{1.f};
inline ModalModes PolarizedMode(vec3 direction, float freq, float t60) {
    ModalModes modes;
    modes.Freqs.push_back(freq);
    modes.T60s.push_back(t60);
    SampleStrip(modes);
    for (auto &shape : modes.Shapes) shape.push_back(SeatedModeShape * direction);
    return modes;
}

constexpr float SeatedCorrelation{5e-5f}, SeatedSlope{-2.4f};
constexpr float SeatedCutoff{SeatedCorrelation / PresetBandRatio};
inline float SeatedSpacing() { return FinishTrackSpacing(SeatedCutoff); }
constexpr float SeatedSigma{2e-6f}; // One face's roughness.
constexpr float SeatedCompositeSigma{SeatedSigma * std::numbers::sqrt2_v<float>};
constexpr double SeatedInvModulus{1.2e-11}; // A ceramic pair's 1/E*
constexpr double SeatedEngagementMax{6 * SeatedCompositeSigma};
constexpr uint32_t SeatedElementColumns{16}, SeatedKnots{56}; // Elements of a quarter correlation length.

inline SubCutoffRoughness SeatedSubCutoff() {
    return UnresolvedBand(SeatedCompositeSigma, SeatedCorrelation, SeatedCutoff, SeatedSlope);
}

inline std::shared_ptr<const ContactSpringSet> MakeSpringSet() {
    // Deterministic and load-independent, so every case presses the one set rather than rebuilding it.
    static const std::shared_ptr<const ContactSpringSet> shared = [] {
        constexpr uint32_t Columns = 1024, Realizations = 8;
        const float spacing = SeatedSpacing();
        const auto rows = uint32_t(4 * SeatedCorrelation / spacing);
        const std::vector<float> gap(rows, 0.f);
        ElementSummits summits;
        std::vector<float> heights(size_t(Columns) * rows);
        for (uint32_t rep = 0; rep < Realizations; ++rep) {
            const auto patch = SynthesizeRoughnessPatch(SeatedCorrelation, SeatedSlope, SeatedCutoff, spacing, Columns, rows, rep);
            for (size_t at = 0; at < heights.size(); ++at) heights[at] = SeatedCompositeSigma * patch.Heights[at];
            GatherElementSummits(summits, heights, Columns, rows, gap, {}, SeatedElementColumns, spacing, SeatedInvModulus);
        }
        auto set = std::make_shared<ContactSpringSet>();
        set->SubCutoff = SeatedSubCutoff();
        set->Springs = BuildElementSprings(summits, set->SubCutoff, SeatedEngagementMax, SeatedKnots);
        set->Curvature = 0;
        set->Reach = std::max(set->Springs.Count() / 2, 1u);
        set->Envelope = ElementEnvelope(set->Springs, set->Curvature, set->Reach);
        FillAnchorForce(*set);
        return set;
    }();
    return shared;
}

constexpr float SeatedLoad{5.f};

inline const RoughnessTrack &SeatedFinishTrack() {
    static const RoughnessTrack track = SynthesizeRoughness(SeatedCorrelation, SeatedSlope, SeatedCutoff, SeatedSpacing(), TrackSamples);
    return track;
}

inline SustainedState SeatedSpringContact(int32_t spring_slot, const ContactSpringSet &set, float flank_stiffness, float load) {
    SustainedState e;
    e.Blend = {.Points = {0, 1, 0}, .Weights = {0.5f, 0.5f, 0.f}};
    e.N = ContactNormal;
    e.NormalForce = load;
    e.Friction = 0.5f;
    e.DampingFactor = 1.5f; // Restitution zero, the ceiling PressedRing must author to seat at all.
    e.StaticPenetration = SolveSpringAnchor(set, load);
    e.SpringIndex = spring_slot;
    e.FlankStiffness = flank_stiffness;
    return e;
}

inline float SeatedShearStiffness(float load) { return float(0.82 * double(load) / (0.4 * double(SeatedCompositeSigma))); }

inline SustainedState SeatedTangentContact(int32_t spring_slot, const ContactSpringSet &set, float load) {
    auto e = SeatedSpringContact(spring_slot, set, 0.f, load);
    e.SlipDir = ContactTangent;
    e.ShearStiffness = SeatedShearStiffness(load);
    return e;
}

struct SeatedFlankRead {
    double SineCube{0};
    double Modulus{0}; // N/m^2, per element, as the resolve's own walk averages it.
};
inline SeatedFlankRead SeatedFlank(const ContactSpringSet &set, float load) {
    const auto [moments, sampled] = SpringFlankSum(set, SolveSpringAnchor(set, load), 1);
    const auto &track = SeatedFinishTrack();
    const double tilt = double(SeatedSigma) * TrackSlopeAtWidth(track, float(SpringBearingWidth(moments) / track.Spacing)) / double(track.Spacing);
    return {FlankSineCube(tilt), moments.Modulus / set.Springs.Count()};
}

inline float SeatedFlankStiffness(const ContactSpringSet &set, float load) {
    const auto read = SeatedFlank(set, load);
    return float(FlankJunctionStiffness(read.SineCube, double(load), read.Modulus));
}

constexpr uint32_t RingWindowSamples{1200}; // 25 ms, the pressed-ring envelope's own window
constexpr uint32_t RingBlocks{94}; // About one second past the strike

// One window's fluctuation level, dB.
// The anchor solves the stack's mean force over positions and the contact sits at one of them, so removing the window's own mean leaves the ring alone.
inline double WindowLevelDb(std::span<const float> w) {
    double mean = 0;
    for (const float s : w) mean += double(s);
    mean /= double(w.size());
    double energy = 0;
    for (const float s : w) energy += (double(s) - mean) * (double(s) - mean);
    return 10.0 * std::log10(std::max(energy / double(w.size()), 1e-300));
}

inline std::vector<float> SeatedStrikeRender(const ModalModes &modes, const std::shared_ptr<const ContactSpringSet> &springs, auto &&make_state, float impulse, uint32_t blocks) {
    Scene scene{modes, 1, 1, 0.f};
    BeginSurfaceTrackFrame(Surface(scene.Audio));
    const auto slot = AdoptContactSprings(Surface(scene.Audio), 2, [&springs] { return springs; });
    const auto seated = make_state(slot);
    scene.Render(4, BlockSize, &seated);
    scene.Strike(impulse);
    return scene.Render(blocks, BlockSize, &seated);
}

// The fluctuation level in each window of one struck arm, dB.
inline std::vector<double> SeatedRingLevels(const ModalModes &modes, const std::shared_ptr<const ContactSpringSet> &springs, auto &&make_state, float impulse) {
    const auto signal = SeatedStrikeRender(modes, springs, make_state, impulse, RingBlocks);
    std::vector<double> levels;
    for (size_t at = 0; at + RingWindowSamples <= signal.size(); at += RingWindowSamples) {
        levels.push_back(WindowLevelDb({signal.data() + at, RingWindowSamples}));
    }
    return levels;
}

struct RingFit {
    double Exponent{0}; // n in rate ~ A^(n-2), from the local rate's slope against level
    double Rate{0}; // The contact-borne rate at the loudest fitted window, dB/s
    double Level{0}; // That window's own level, dB
    uint32_t Points{0};
};

inline double RingRateAt(const RingFit &fit, double level) {
    return fit.Points > 0 ? fit.Rate * std::pow(10.0, (fit.Exponent - 2.0) * (level - fit.Level) / 20.0) : 0.0;
}

inline RingFit FitRing(const std::vector<double> &levels, double material_rate, RingFit baseline = {}) {
    constexpr size_t Onset{4};
    const double window_seconds = double(RingWindowSamples) / double(SampleRate);
    double peak = -1e300;
    for (size_t i = Onset; i < levels.size(); ++i) peak = std::max(peak, levels[i]);
    RingFit fit;
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (size_t i = Onset; i + 1 < levels.size(); ++i) {
        if (levels[i] < peak - 40) break;
        const double rate = (levels[i] - levels[i + 1]) / window_seconds - material_rate - RingRateAt(baseline, levels[i]);
        if (!(rate > 0)) continue;
        // Level is an amplitude in dB, so a decade of amplitude is 20 dB and the slope is n-2.
        const double x = levels[i] / 20, y = std::log10(rate);
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
        if (fit.Points == 0) {
            fit.Rate = rate;
            fit.Level = levels[i];
        }
        ++fit.Points;
    }
    const double n = double(fit.Points);
    const double denom = n * sxx - sx * sx;
    fit.Exponent = fit.Points > 2 && std::abs(denom) > 0 ? 2.0 + (n * sxy - sx * sy) / denom : 0.0;
    return fit;
}
