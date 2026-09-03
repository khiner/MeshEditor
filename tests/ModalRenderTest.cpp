
#include "ModalBench.h"
#include "Near.h"
#include "RunSuites.h"

#include <boost/ut.hpp>

#include <array>
#include <cmath>
#include <numbers>
#include <span>
#include <vector>

using namespace boost::ut;

int main() {
    "excitations superpose linearly"_test = [] {
        const auto render = [](std::span<const ModalEvent> events) {
            ModalScene scene{1, 64, 0.2f, 1};
            for (auto e : events) {
                e.Object = scene.Objects.front();
                EnqueueModalEvent(scene.Audio, e);
            }
            return scene.Render(8, BlockSize);
        };
        const std::array both{ImpactEvent(0, 1.f, 0, 1.f / 300.f), ImpactEvent(0, -0.4f, 1, 1.f / 90.f)};
        const auto a = render(std::span{both}.first(1)), b = render(std::span{both}.last(1)), together = render(both);
        std::vector<float> sum(a.size());
        for (size_t i = 0; i < a.size(); ++i) sum[i] = a[i] + b[i];
        expect(Peak(a) > 0._f);
        expect(Peak(b) > 0._f);
        expect(MaxDifference(together, sum) <= Peak(together) * 1e-5) << MaxDifference(together, sum);
    };

    "a strike does not depend on how many threads share it"_test = [] {
        const auto render = [](uint32_t renderers) {
            ModalScene scene{16, 64, 0.2f, renderers};
            for (const auto o : scene.Objects) EnqueueModalEvent(scene.Audio, ImpactEvent(o, 1.f));
            return scene.Render(32, BlockSize);
        };
        const auto single = render(1), split = render(4);
        expect(Peak(single) > 0.f);
        expect(MaxDifference(single, split) < Peak(single) * 1e-5f);
    };

    // The click is the rigid recoil the contact force radiates, discretized at the bank rate.
    // Its physical peak belongs to the collision, so oversampling the render must not move it.
    "the acceleration-noise click does not depend on the output sample rate"_test = [] {
        constexpr double Tau{5e-4}; // seconds, the same physical contact whatever the rate renders it
        constexpr double Radius{0.05}, Volume{4.0 / 3.0 * std::numbers::pi * Radius * Radius * Radius}, Mass{1.0}, Impulse{0.5};
        const auto peak_at = [](float rate) {
            ModalScene scene{1, 64, 0.2f, 1, 0.f, rate};
            const auto step = float(1.0 / (Tau * double(rate)));
            const auto click = RecoilClickFilter(Radius, Volume, Mass, rate);
            // No impulse on the modes, so they stay silent and the click is the whole output.
            EnqueueModalEvent(scene.Audio, {.Kind = ModalEventKind::Impact, .Object = scene.Objects.front(), .ExPos = 0, .Jx = 0.f, .Jy = 0.f, .Jz = 0.f, .PulseStep = step, .PulseGamma = 2 * step, .AccelAmp = float(Impulse) * rate, .ClickB0 = click.B0, .ClickA1 = click.A1, .ClickA2 = click.A2});
            const auto blocks = uint32_t(std::ceil(4 * Tau * double(rate) / BlockSize));
            return Peak(scene.Render(blocks, BlockSize));
        };
        const auto slow = peak_at(SampleRate), fast = peak_at(2 * SampleRate);
        expect(slow > 0._f);
        expect(Near(double(fast) / double(slow), 1.0, 2e-2)) << slow << fast;
    };

    return RunSuites();
}
