#include "LoadObj.h"
#include "audio/AcousticMaterial.h"
#include "audio/RealImpact.h"
#include "audio/mesh2modes.h"
#include "mesh/Tets.h"

#include "tetgen.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <print>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {
// A spread of sizes and materials. Any dataset sample can be named on the command line instead.
constexpr std::string_view DefaultSamples[]{
    "9_BowlCeramic",
    "22_Cup",
    "19_Pan",
    "27_WoodPlate",
    "50_PlasticBowl",
    "91_MetalSpoon",
    "35_MeasuringCup",
    "74_FlowerPotLargeCeramic",
};

constexpr uint32_t NumExcitePositions{10}; // Matches the app's default excitable vertex count.
constexpr modal::SolverConfig SolveConfig{};

struct BenchResult {
    std::string Name, Material;
    size_t SurfaceTris, TetCount;
    modal::SolveProfile Profile;
    double LoadSeconds, TetsSeconds;
    size_t NumModes;
    double FundamentalHz;

    double Total() const {
        const auto &p = Profile;
        return LoadSeconds + TetsSeconds + p.MassProps + p.QuadMesh + p.Assemble + p.SampleExcite + p.Factorize + p.Iterate + p.Extract;
    }
};

double SecondsSince(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

const AcousticMaterial &MaterialForSample(const std::string &sample_dir) {
    if (const auto object_name = RealImpact::FindObjectName(fs::path{sample_dir})) {
        if (const auto material_name = RealImpact::FindMaterialName(*object_name)) {
            for (const auto &m : materials::acoustic::All) {
                if (m.Name == *material_name) return m;
            }
        }
    }
    std::println("  (no material mapping for '{}', using {})", sample_dir, materials::acoustic::All.front().Name);
    return materials::acoustic::All.front();
}

// One block per sample: geometry and profile lines, then one line per mode.
// Gain is the RMS shape magnitude across excitation positions (mode shape signs are arbitrary).
void PrintCorpusSample(const BenchResult &r, const modal::ModalResult &result) {
    const auto &p = r.Profile;
    std::println("=== {} ({}) ===", r.Name, r.Material);
    std::println("geometry: {} tris -> {} tets, {} dofs, {} nnz(K)", r.SurfaceTris, r.TetCount, p.Dofs, p.StiffnessNonZeros);
    std::println("profile: tetgen {:.2f} s, assemble {:.2f} s, factorize {:.2f} s, iterate {:.2f} s (opsolve {:.2f} s), {} ops, {} restarts", r.TetsSeconds, p.Assemble, p.Factorize, p.Iterate, p.OpSolve, p.OpApplications, p.Restarts);
    std::println("mass: {:.4g} kg", result.MassProps.Mass);
    const auto &modes = result.Modes;
    std::println("modes: {}", modes.Freqs.size());
    for (size_t m = 0; m < modes.Freqs.size(); ++m) {
        double gain_sq = 0;
        for (const auto &shape : modes.Shapes) gain_sq += double(glm::dot(shape[m], shape[m]));
        const double gain = std::sqrt(gain_sq / modes.Shapes.size());
        std::println("  {:>2}: {:>9.2f} Hz, t60 {:>7.3f} s, gain {:.3e}", m + 1, modes.Freqs[m], modes.T60s[m], gain);
    }
    std::println("");
}

std::optional<BenchResult> RunSample(const fs::path &dataset, const std::string &name, float ratio, bool corpus) {
    const auto obj_path = dataset / name / "preprocessed" / "transformed.obj";
    if (!fs::exists(obj_path)) {
        std::println("skipping {}: {} not found", name, obj_path.string());
        return {};
    }

    BenchResult r{.Name = name};
    const auto &material = MaterialForSample(name);
    r.Material = material.Name;

    auto load_start = std::chrono::steady_clock::now();
    auto surface = LoadObj(obj_path);
    r.LoadSeconds = SecondsSince(load_start);
    if (!surface || surface->Positions.empty()) {
        std::println("skipping {}: failed to load {}", name, obj_path.string());
        return {};
    }
    r.SurfaceTris = surface->TriangleIndices.size() / 3;

    // Excitation positions spread evenly across surface vertices, like the app's default.
    const auto num_verts = surface->Positions.size();
    std::vector<vec3> excite_positions(NumExcitePositions);
    for (uint32_t i = 0; i < NumExcitePositions; ++i) excite_positions[i] = surface->Positions[i * num_verts / NumExcitePositions];

    // tetgen signals failure (e.g. a self-intersecting surface) by throwing an int.
    try {
        const auto tets_start = std::chrono::steady_clock::now();
        const auto tets = GenerateTets(std::move(surface->Positions), std::move(surface->TriangleIndices), {.PreserveSurface = true, .SimplifyRatio = ratio});
        r.TetsSeconds = SecondsSince(tets_start);
        r.TetCount = tets->numberoftetrahedra;

        const auto result = modal::mesh2modes(*tets, material.Properties, excite_positions, vec3{1}, SolveConfig);
        r.Profile = result.Profile;
        r.NumModes = result.Modes.Freqs.size();
        r.FundamentalHz = r.NumModes > 0 ? result.Modes.Freqs.front() : 0;
        if (corpus) PrintCorpusSample(r, result);
        return r;
    } catch (const std::exception &e) {
        std::println("skipping {}: solve failed: {}", name, e.what());
    } catch (int code) {
        std::println("skipping {}: tetgen failed with code {}", name, code);
    }
    return {};
}

void PrintHeader() {
    std::println("{:<26} {:>8} {:>8} {:>8} {:>8} {:>11} | {:>6} {:>7} {:>6} {:>6} {:>7} {:>8} {:>8} {:>8} {:>6} {:>6} {:>8} | {:>5} {:>5} {:>4} {:>9}", "sample", "material", "tris", "tets", "dofs", "nnz(K)", "load", "tetgen", "mass", "quad", "asm", "factor", "iter", "opsolve", "excite", "extr", "total", "modes", "ops", "rst", "f1");
}

void PrintRow(const BenchResult &r) {
    const auto &p = r.Profile;
    std::println("{:<26} {:>8} {:>8} {:>8} {:>8} {:>11} | {:>6.2f} {:>7.2f} {:>6.2f} {:>6.2f} {:>7.2f} {:>8.2f} {:>8.2f} {:>8.2f} {:>6.2f} {:>6.2f} {:>8.2f} | {:>5} {:>5} {:>4} {:>9.1f}", r.Name, r.Material, r.SurfaceTris, r.TetCount, p.Dofs, p.StiffnessNonZeros, r.LoadSeconds, r.TetsSeconds, p.MassProps, p.QuadMesh, p.Assemble, p.Factorize, p.Iterate, p.OpSolve, p.SampleExcite, p.Extract, r.Total(), r.NumModes, p.OpApplications, p.Restarts, r.FundamentalHz);
}

void PrintSummary(const std::vector<BenchResult> &results, float ratio) {
    double load = 0, tets = 0;
    modal::SolveProfile sum;
    for (const auto &r : results) {
        load += r.LoadSeconds;
        tets += r.TetsSeconds;
        sum += r.Profile;
    }
    const double grand = load + tets + sum.MassProps + sum.QuadMesh + sum.Assemble + sum.SampleExcite + sum.Factorize + sum.Iterate + sum.Extract;
    if (grand <= 0) return;

    std::println("\n--- totals across {} samples, simplify ratio {:.2f}: {:.2f} s ---", results.size(), ratio, grand);
    const auto line = [&](std::string_view section, double seconds) {
        std::println("{:>10}: {:>8.2f} s {:>5.1f}%", section, seconds, 100 * seconds / grand);
    };
    line("load", load);
    line("tetgen", tets);
    line("mass", sum.MassProps);
    line("quad", sum.QuadMesh);
    line("assemble", sum.Assemble);
    line("factorize", sum.Factorize);
    line("iterate", sum.Iterate);
    std::println("{:>10}: {:>8.2f} s ({:.1f}% of iterate)", "opsolve", sum.OpSolve, sum.Iterate > 0 ? 100 * sum.OpSolve / sum.Iterate : 0);
    line("excite", sum.SampleExcite);
    line("extract", sum.Extract);
}
} // namespace

int main(int argc, char **argv) {
    const char *env_dataset = std::getenv("REALIMPACT_DATASET_DIR");
    fs::path dataset = env_dataset ? env_dataset : REALIMPACT_DATASET_DIR;
    float ratio = 1;
    bool all = false;
    bool corpus = false;
    std::vector<std::string> names;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg{argv[i]};
        if (arg == "--all") all = true;
        else if (arg == "--corpus") corpus = true;
        else if (arg == "--ratio" && i + 1 < argc) ratio = std::stof(argv[++i]);
        else if (arg == "--dataset" && i + 1 < argc) dataset = argv[++i];
        else names.emplace_back(arg);
    }
    if (!fs::is_directory(dataset)) {
        std::println("dataset directory not found: {}", dataset.string());
        return 1;
    }
    if (names.empty()) {
        // Corpus mode sweeps the whole dataset: its output tracks solver changes across commits.
        if (all || corpus) {
            for (const auto &entry : fs::directory_iterator{dataset}) {
                if (entry.is_directory() && fs::exists(entry.path() / "preprocessed" / "transformed.obj")) names.push_back(entry.path().filename().string());
            }
            std::ranges::sort(names, {}, [](const std::string &n) { return std::atoi(n.c_str()); });
        } else {
            names.assign(std::begin(DefaultSamples), std::end(DefaultSamples));
        }
    }

    if (corpus) {
        std::println("modal solve corpus: {} samples, simplify ratio {:.2f}, {} excitation positions", names.size(), ratio, NumExcitePositions);
        std::println("solver config: modes {:g}-{:g} Hz, keep {} of {} eigenpairs, tolerance {:g}, max restarts {}", SolveConfig.MinModeFreq, SolveConfig.MaxModeFreq, SolveConfig.NumModes, SolveConfig.NumFemModes, SolveConfig.Tolerance, SolveConfig.MaxRestarts);
        std::println("");
    } else {
        PrintHeader();
    }
    std::vector<BenchResult> results;
    for (const auto &name : names) {
        if (auto result = RunSample(dataset, name, ratio, corpus)) {
            if (!corpus) PrintRow(*result);
            results.push_back(std::move(*result));
        }
    }
    PrintSummary(results, ratio);
    return results.empty() ? 1 : 0;
}
