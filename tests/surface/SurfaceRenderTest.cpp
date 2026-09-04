
#include "SeatedBench.h"

#include "Near.h"
#include "RunSuites.h"

#include <boost/ut.hpp>

#include <cmath>
#include <numbers>
#include <span>
#include <vector>

using namespace boost::ut;

namespace {

// A contact at rest must reproduce k * delta0^(3/2) == N exactly, with no rounding.
// Powers of two give that: 2^-18 m under 2^31 N/m^(3/2) meets 16 N.
constexpr float RestPenetration{0x1p-18f}, RestStiffness{0x1p31f}, RestLoad{0x1p4f};

// A contact travelling over both surfaces, as a scrape does.
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
    e.DampingFactor = 0.4f;
    for (auto &t : e.Tracks) t = {.Index = slot, .Rate = 0.4f, .Sigma = 2e-7f, .Window = 8.f, .Step = 4e-7f};
    return e;
}

SustainedState RestingContact(int32_t slot) {
    auto e = MovingContact(slot);
    e.SlipDir = vec3{0};
    e.SweepDir = {vec3{0}, vec3{0}};
    for (auto &t : e.Tracks) t = {.Index = slot, .Rate = 0.f, .Sigma = 2e-7f, .Window = 8.f, .Step = 0.f};
    return e;
}

// Spread of asperity heights about their own cell's local mean, m.
constexpr float RestCellSpread{0x1p-18f};

SustainedState AsBed(SustainedState e, float cell_spread = RestCellSpread) {
    e.Stiffness = 0.f;
    e.SpotCount = 8;
    e.SpotWeight = 4.f;
    e.SpotStiffness = RestStiffness;
    e.CellSpread = cell_spread;
    e.StaticPenetration = RestPenetration;
    const float scale = e.SpotWeight * e.SpotStiffness * e.CellSpread * std::sqrt(e.CellSpread);
    e.NormalForce = float(e.SpotCount) * scale * BedLoadFactor(e.StaticPenetration / e.CellSpread);
    return e;
}

void ExpectSettles(auto &&make_state) {
    Scene scene{1, 64, 0.2f, 1};
    const auto state = make_state(scene.Slot);
    const auto signal = scene.Render(256, BlockSize, &state);
    const std::span whole{signal};
    const auto quarter = whole.size() / 4;
    expect(AllFinite(whole));
    expect(Peak(whole.subspan(2 * quarter, quarter)) > 0.f);
    expect(Peak(whole.last(quarter)) < Peak(whole.subspan(2 * quarter, quarter)) * 2.f);
}

// A contact pressing with nothing, which drives no mode however the force model is written.
SustainedState SilentContact() {
    SustainedState e;
    e.Blend = {.Points = {0, 1, 0}, .Weights = {0.5f, 0.5f, 0.f}};
    e.N = {0.f, 1.f, 0.f};
    return e;
}

double RelaxationPerCycle(float scale, float approach_amp, float tangent_amp, double phase, double per_cycle, uint32_t cycles = 40) {
    SustainedState st;
    st.RelaxScale = scale;
    // A real coefficient, since a contact carrying none is not one this model renders.
    // The channel does not read it, so the closed-form checks below are unaffected.
    st.Friction = 0.5f;
    SustainedCarry carry;
    const auto depth = [=](uint32_t i) { return approach_amp * std::cos(2 * std::numbers::pi * i / per_cycle); };
    const auto tangent = [=](uint32_t i) { return tangent_amp * std::cos(2 * std::numbers::pi * i / per_cycle + phase); };
    const auto settle = uint32_t(2 * per_cycle);
    double shed = 0;
    for (uint32_t i = 0; i < uint32_t(cycles * per_cycle); ++i) {
        const float released = RelaxationRelease(st, carry, float(depth(i + 1) - depth(i)), float(tangent(i + 1) - tangent(i)));
        if (i >= settle) shed += released;
    }
    return shed / (cycles - settle / per_cycle);
}
} // namespace
int main() {
    "a contact at rest excites nothing"_test = [] {
        Scene scene{1, 64, 0.2f, 1};
        const auto resting = RestingContact(scene.Slot);
        expect(Peak(scene.Render(8, BlockSize, &resting)) == 0.f);
    };

    "a bedded contact at rest excites nothing"_test = [] {
        Scene scene{1, 64, 0.2f, 1};
        const auto resting = AsBed(RestingContact(scene.Slot));
        expect(Peak(scene.Render(8, BlockSize, &resting)) == 0.f);
    };

    "a bed whose cells are one asperity wide takes Hertz outright"_test = [] {
        Scene scene{1, 64, 0.2f, 1};
        auto state = AsBed(MovingContact(scene.Slot));
        state.CellSpread = 0.f;
        state.NormalForce = RestLoad;
        const auto signal = scene.Render(64, BlockSize, &state);
        expect(AllFinite(signal));
        expect(Peak(signal) > 0.f);
    };

    "the bed's height spread reaches the render"_test = [] {
        const auto render = [](float spread) {
            Scene scene{1, 64, 0.2f, 1};
            const auto state = AsBed(MovingContact(scene.Slot), spread);
            return scene.Render(64, BlockSize, &state);
        };
        const auto stiff = render(RestCellSpread), soft = render(4.f * RestCellSpread);
        expect(AllFinite(stiff));
        expect(AllFinite(soft));
        expect(Peak(stiff) > 0.f);
        expect(MaxDifference(stiff, soft) > 0.f);
    };

    "a moving bedded contact settles rather than diverging"_test = [] {
        ExpectSettles([](int32_t slot) { return AsBed(MovingContact(slot)); });
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
        // Five T60s put a mode's ring-down at 1e-15 of its peak, and the exchange's state flushes to a float dust floor near 1e-12 absolute.
        // A loop feeding itself sits decades above either (a self-oscillation rails at 1e4 times the open loop), so the bound sits over the dust floor.
        expect(Peak(whole.last(BlockSize)) < Peak(whole) * 1e-8f) << Peak(whole) << Peak(whole.last(BlockSize));
    };

    "a moving contact settles rather than diverging"_test = [] { ExpectSettles(MovingContact); };

    "coupling damps the contact rather than driving it"_test = [] {
        const auto render = [](float coupling) {
            Scene scene{1, 64, 0.2f, 1};
            Surface(scene.Audio).Coupling.store(coupling);
            const auto moving = MovingContact(scene.Slot);
            return Rms(scene.Render(256, BlockSize, &moving));
        };
        const auto open_loop = render(0.f), coupled = render(1.f);
        expect(open_loop > 0.);
        expect(coupled < open_loop);
    };

    "a silent contact leaves a strike alone"_test = [] {
        const auto render = [](bool voice) {
            Scene scene{1, 200, 0.2f, 1};
            scene.Strike(1.f);
            const auto silent = SilentContact();
            return scene.Render(16, BlockSize, voice ? &silent : nullptr);
        };
        const auto without = render(false), with = render(true);
        expect(Peak(without) > 0.f);
        // The radiation gains span the band's (ka) range, so the kernels' summation orders sit further apart than under a uniform per-mode scale.
        // Measured 1.2e-4 of peak, identical with coupling off.
        expect(MaxDifference(without, with) < Peak(without) * 5e-4f);
    };

    "coupling saturates against solved-scale shapes"_test = [] {
        const auto render = [](float coupling, uint32_t blocks) {
            Scene scene{1, 64, 20.f, 1, 0.f, SampleRate, 200.f};
            Surface(scene.Audio).Coupling.store(coupling);
            const auto moving = AsBed(MovingContact(scene.Slot));
            return scene.Render(blocks, BlockSize, &moving);
        };
        const auto open_loop = render(0.f, 64);
        const auto coupled = render(1.f, 512);
        expect(AllFinite(coupled));
        expect(Peak(open_loop) > 0.f);
        const auto q = coupled.size() / 4;
        const auto r3 = Rms({coupled.data() + 2 * q, q}), r4 = Rms({coupled.data() + 3 * q, q});
        expect(r4 < r3 * 1.5) << r3 << r4;
    };

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

    static constexpr float RestInvMass{1.6f};

    "a contact at rest excites nothing when the body can move"_test = [] {
        Scene scene{1, 64, 0.2f, 1, RestInvMass};
        const auto resting = RestingContact(scene.Slot);
        const auto signal = scene.Render(8, BlockSize, &resting);
        expect(Peak(signal) == 0._f);
    };

    "the body's own motion changes what a contact excites"_test = [] {
        Scene fixed{1, 64, 0.2f, 1, 0.f};
        Scene movable{1, 64, 0.2f, 1, RestInvMass};
        const auto fixed_contact = MovingContact(fixed.Slot);
        const auto movable_contact = MovingContact(movable.Slot);
        const auto a = fixed.Render(8, BlockSize, &fixed_contact);
        const auto b = movable.Render(8, BlockSize, &movable_contact);
        expect(Peak(a) > 0._f);
        expect(MaxDifference(a, b) > 0._f);
    };

    // The contact resists the body rather than driving it, so the body stays near its equilibrium instead of drifting away from it.
    // The bounded excitation keeps the audio finite either way, so only the body's own displacement separates a restoring contact from a runaway one.
    "the body rides its contact rather than walking off it"_test = [] {
        Scene scene{1, 64, 0.2f, 1, RestInvMass};
        const auto moving = MovingContact(scene.Slot);
        scene.Render(64, BlockSize, &moving);
        const auto travel = std::abs(Surface(scene.Audio).VoiceCarry.front().RigidNormal);
        expect(travel < 10.f * RestPenetration) << travel;
    };

    // A contact whose geometry changes under it reports an approach from far outside the regime either force law describes, and both grow without bound in it.
    // Hertz's power law overflows there, and the render stays finite through it rather than turning the whole scene's audio to NaN.
    "a contact driven far past its equilibrium stays finite"_test = [] {
        Scene scene{1, 64, 0.2f, 1, RestInvMass};
        auto wild = MovingContact(scene.Slot);
        for (auto &t : wild.Tracks) t.Sigma = 1e24f;
        const auto signal = scene.Render(8, BlockSize, &wild);
        expect(AllFinite(signal));
    };

    "a bedded contact holds the body inside its own height spread"_test = [] {
        Scene scene{1, 64, 0.2f, 1, RestInvMass};
        const auto moving = AsBed(MovingContact(scene.Slot));
        scene.Render(64, BlockSize, &moving);
        const auto travel = std::abs(Surface(scene.Audio).VoiceCarry.front().RigidNormal);
        expect(travel < 2.f * RestCellSpread) << travel;
    };

    "a contact whose normal turns rides it as steadily as a fixed one"_test = [] {
        constexpr uint32_t Blocks{64};
        struct Result {
            float Travel;
            double Level;
        };
        const auto run = [](float turn_per_block) {
            Scene scene{1, 64, 0.2f, 1, RestInvMass};
            std::vector<float> signal(size_t(Blocks) * BlockSize, 0.f);
            for (uint32_t block = 0; block < Blocks; ++block) {
                const float angle = turn_per_block * float(block);
                const float c = std::cos(angle), s = std::sin(angle);
                auto state = MovingContact(scene.Slot);
                state.N = {s, c, 0.f};
                state.SlipDir = {c, -s, 0.f};
                state.SweepDir[0] = {c, -s, 0.f};
                scene.Publish(state);
                RenderModal(scene.Audio, signal.data() + size_t(block) * BlockSize, BlockSize);
            }
            expect(AllFinite(signal));
            return Result{std::abs(Surface(scene.Audio).VoiceCarry.front().RigidNormal), Rms(signal)};
        };
        // One full turn over the render, which a body a few centimetres across rolling at a walk makes.
        const auto rolling = run(2.f * std::numbers::pi_v<float> / float(Blocks)), fixed = run(0.f);
        expect(rolling.Travel < 10.f * RestPenetration) << rolling.Travel;
        expect(rolling.Level < fixed.Level * 2) << rolling.Level << fixed.Level;
    };

    "a movable body settles rather than diverging"_test = [] {
        Scene scene{1, 64, 0.2f, 1, RestInvMass};
        const auto moving = MovingContact(scene.Slot);
        const auto signal = scene.Render(64, BlockSize, &moving);
        expect(AllFinite(signal));
        const std::span<const float> whole{signal};
        expect(Rms(whole.subspan(whole.size() / 2)) <= Rms(whole.subspan(0, whole.size() / 2)) * 4.0);
    };

    "a flank junction shortens a pressed mode's ring"_test = [] {
        const auto springs = MakeSpringSet();
        const float flank = SeatedFlankStiffness(*springs, SeatedLoad);
        expect(flank > 0._f) << flank;
        const auto ring_rate = [&springs](float flank_stiffness, float impulse) {
            const auto signal = SeatedStrikeRender(
                PolarizedMode(ContactNormal, 5000.f, 10.f), springs,
                [&springs, flank_stiffness](int32_t slot) { return SeatedSpringContact(slot, *springs, flank_stiffness, SeatedLoad); },
                impulse, 64
            );
            expect(AllFinite(signal));
            const std::span<const float> whole{signal};
            const auto window = whole.size() / 8;
            const auto early = WindowLevelDb(whole.subspan(window, window)), late = WindowLevelDb(whole.subspan(6 * window, window));
            return (late - early) / (5.0 * double(window) / double(SampleRate));
        };
        const auto bare = ring_rate(0.f, 0.1f), joined = ring_rate(flank, 0.1f);
        expect(bare < 0.0) << bare;
        expect(joined < bare) << bare << joined;
    };

    "a loaded tangential junction dissipates more at a heavier load"_test = [] {
        const auto springs = MakeSpringSet();
        constexpr float RingFreq{5000}, RingT60{10};
        const auto junction_rate = [&springs](float load) {
            const auto fit = [&](bool frictional) {
                const auto levels = SeatedRingLevels(
                    PolarizedMode(numeric::Normalize(ContactNormal + ContactTangent), RingFreq, RingT60), springs,
                    [&springs, load, frictional](int32_t slot) {
                        auto seated = SeatedTangentContact(slot, *springs, load);
                        if (!frictional) seated.Friction = 0.f;
                        return seated;
                    },
                    0.1f
                );
                return FitRing(levels, 60.0 / RingT60);
            };
            return fit(true).Rate - fit(false).Rate;
        };
        const auto light = junction_rate(1.25f), heavy = junction_rate(20.f);
        expect(light > 0.0) << "the junction dissipates at all" << light;
        expect(heavy > light) << "tangential damping rises with normal load" << light << heavy;
    };

    "the relaxation channel reproduces the loss it integrates"_test = [] {
        constexpr float Scale{1}, Amplitude{1e-8f};
        // A 5 kHz mode at the render rate, and a fine sampling the closed form is exact at.
        for (const double per_cycle : {9.6, 96.0}) {
            const auto shed = [per_cycle](double phase, float approach_amp, float tangent_amp) {
                return RelaxationPerCycle(Scale, approach_amp, tangent_amp, phase, per_cycle);
            };
            const auto closed_form = [](double phase, float approach_amp, float tangent_amp) {
                const double s = std::sin(phase);
                return 16.0 / 3.0 * Scale * double(tangent_amp) * tangent_amp * std::abs(approach_amp) * s * s;
            };
            const double quadrature = shed(std::numbers::pi / 2, Amplitude, Amplitude);
            // One shared phase releases nothing, at any sampling.
            expect(shed(0, Amplitude, Amplitude) < 1e-6 * quadrature);
            // The magnitude, which the coarse sampling of one cycle's turning points sets the floor of.
            expect(quadrature > 0.85 * closed_form(std::numbers::pi / 2, Amplitude, Amplitude));
            expect(quadrature < 1.02 * closed_form(std::numbers::pi / 2, Amplitude, Amplitude));
            // The phase law, and the amplitude law: the tangential amplitude squared, the normal once.
            for (const double phase : {0.1, 0.5, 1.0}) {
                expect(Near(shed(phase, Amplitude, Amplitude) / quadrature, std::pow(std::sin(phase), 2), 0.01));
            }
            expect(Near(shed(std::numbers::pi / 2, Amplitude, 3 * Amplitude) / quadrature, 9., 0.01));
            expect(Near(shed(std::numbers::pi / 2, 3 * Amplitude, Amplitude) / quadrature, 3., 0.01));
        }
    };

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

    return RunSuites();
}
