
#include "LoadObj.h"
#include "audio/AcousticMaterialProperties.h"
#include "audio/mesh2modes.h"
#include "mesh/Tets.h"

#include <charconv>
#include <cstdio>
#include <numbers>
#include <print>
#include <string_view>

namespace {
double ArgValue(int argc, char **argv, std::string_view name, double fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            double value;
            const std::string_view s = argv[i + 1];
            if (std::from_chars(s.data(), s.data() + s.size(), value).ec == std::errc{}) return value;
        }
    }
    return fallback;
}

bool HasFlag(int argc, char **argv, std::string_view name) {
    for (int i = 1; i < argc; ++i) {
        if (argv[i] == name) return true;
    }
    return false;
}

void PrintScalars(std::string_view key, const auto &values) {
    std::print("  \"{}\": [", key);
    for (size_t i = 0; i < values.size(); ++i) std::print("{}{}", i ? "," : "", values[i]);
    std::println("],");
}
}
int main(int argc, char **argv) {
    if (argc < 2) {
        std::println(stderr, "Usage: {} <mesh.obj> [--young E] [--poisson v] [--density rho] [--alpha a] [--beta b] [--min-freq f] [--max-freq f] [--modes n] [--quality]", argv[0]);
        return 1;
    }
    const auto mesh = LoadObj(argv[1]);
    if (!mesh || mesh->Positions.empty()) {
        std::println(stderr, "Failed to load mesh: {}", argv[1]);
        return 1;
    }

    const AcousticMaterialProperties material{
        .Density = ArgValue(argc, argv, "--density", 2700),
        .YoungModulus = ArgValue(argc, argv, "--young", 7.2e10),
        .PoissonRatio = ArgValue(argc, argv, "--poisson", 0.19),
        .Alpha = ArgValue(argc, argv, "--alpha", 5),
        .Beta = ArgValue(argc, argv, "--beta", 2e-8),
    };
    const modal::SolverConfig config{
        .MinModeFreq = float(ArgValue(argc, argv, "--min-freq", 20)),
        .MaxModeFreq = float(ArgValue(argc, argv, "--max-freq", 16'000)),
        .NumModes = uint32_t(ArgValue(argc, argv, "--modes", 30)),
        .NumFemModes = uint32_t(ArgValue(argc, argv, "--modes", 30)) + 15,
    };

    auto tets = GenerateTets(mesh->Positions, mesh->TriangleIndices, {.Quality = HasFlag(argc, argv, "--quality")});
    if (!tets) {
        std::println(stderr, "Tetrahedralization failed: {}", tets.error());
        return 1;
    }
    const auto result = modal::mesh2modes(tets->Mesh, material, mesh->Positions, vec3{1}, config);
    const auto &modes = result.Modes;
    if (modes.Freqs.empty()) {
        std::println(stderr, "Solve produced no modes in [{} Hz, {} Hz]", config.MinModeFreq, config.MaxModeFreq);
        return 1;
    }

    // The mesh's triangles, relabeled onto the sample points its vertices became.
    // A triangle whose corners merged onto fewer than three points has no area and is dropped.
    std::vector<uint32_t> indices;
    indices.reserve(mesh->TriangleIndices.size());
    for (size_t t = 0; t + 2 < mesh->TriangleIndices.size(); t += 3) {
        const auto a = result.SamplePointOfExcitation[mesh->TriangleIndices[t]];
        const auto b = result.SamplePointOfExcitation[mesh->TriangleIndices[t + 1]];
        const auto c = result.SamplePointOfExcitation[mesh->TriangleIndices[t + 2]];
        if (a == b || b == c || a == c) continue;
        indices.insert(indices.end(), {a, b, c});
    }

    // A T60 is the time to fall 60 dB, so the decay rate it states is this over it.
    static constexpr float Ln1000 = 3 * std::numbers::ln10_v<float>;
    std::vector<float> decay_rates(modes.T60s.size());
    for (size_t k = 0; k < modes.T60s.size(); ++k) decay_rates[k] = modes.T60s[k] > 0 ? Ln1000 / modes.T60s[k] : 0.f;

    std::println("{{");
    PrintScalars("frequencies", modes.Freqs);
    PrintScalars("decayRates", decay_rates);
    std::print("  \"positions\": [");
    for (size_t i = 0; i < modes.Positions.size(); ++i) {
        const auto &p = modes.Positions[i];
        std::print("{}[{},{},{}]", i ? "," : "", p.x, p.y, p.z);
    }
    std::println("],");
    // Mode-major, matching the model schema: every sample point of mode 0, then of mode 1, and so on.
    std::print("  \"shapes\": [");
    for (size_t k = 0; k < modes.Freqs.size(); ++k) {
        for (size_t i = 0; i < modes.Shapes.size(); ++i) {
            const auto &s = modes.Shapes[i][k];
            std::print("{}[{},{},{}]", k || i ? "," : "", s.x, s.y, s.z);
        }
    }
    std::println("],");
    PrintScalars("indices", indices);
    std::println("  \"mass\": {},", result.MassProps.Mass);
    std::println("  \"centerOfMass\": [{},{},{}],", result.MassProps.CenterOfMass.x, result.MassProps.CenterOfMass.y, result.MassProps.CenterOfMass.z);
    std::println("  \"inertiaDiagonal\": [{},{},{}]", result.MassProps.InertiaDiagonal.x, result.MassProps.InertiaDiagonal.y, result.MassProps.InertiaDiagonal.z);
    std::println("}}");
    return 0;
}
