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

// RMS shape magnitude of one mode across excitation positions (mode shape signs are arbitrary).
double RmsGain(const ModalModes &modes, size_t m) {
    double gain_sq = 0;
    for (const auto &shape : modes.Shapes) gain_sq += double(glm::dot(shape[m], shape[m]));
    return std::sqrt(gain_sq / modes.Shapes.size());
}

// One block per sample: geometry and profile lines, then one line per mode.
void PrintCorpusSample(const BenchResult &r, const modal::ModalResult &result) {
    const auto &p = r.Profile;
    std::println("=== {} ({}) ===", r.Name, r.Material);
    std::println("geometry: {} tris -> {} tets, {} dofs, {} nnz(K)", r.SurfaceTris, r.TetCount, p.Dofs, p.StiffnessNonZeros);
    std::println("profile: tetgen {:.2f} s, assemble {:.2f} s, factorize {:.2f} s, iterate {:.2f} s (opsolve {:.2f} s), {} ops, {} restarts", r.TetsSeconds, p.Assemble, p.Factorize, p.Iterate, p.OpSolve, p.OpApplications, p.Restarts);
    std::println("mass: {:.4g} kg", result.MassProps.Mass);
    const auto &modes = result.Modes;
    std::println("modes: {}", modes.Freqs.size());
    for (size_t m = 0; m < modes.Freqs.size(); ++m) {
        std::println("  {:>2}: {:>9.2f} Hz, t60 {:>7.3f} s, gain {:.3e}", m + 1, modes.Freqs[m], modes.T60s[m], RmsGain(modes, m));
    }
    std::println("");
}

// A sample's preprocessed surface, with excitation positions spread evenly across its
// vertices like the app's default. Prints and returns empty on failure.
struct LoadedSample {
    SurfaceMesh Surface;
    std::vector<vec3> ExcitePositions;
    double LoadSeconds;
};

std::optional<LoadedSample> LoadSample(const fs::path &dataset, const std::string &name) {
    const auto obj_path = dataset / name / "preprocessed" / "transformed.obj";
    const auto load_start = std::chrono::steady_clock::now();
    auto surface = LoadObj(obj_path);
    const double load_seconds = SecondsSince(load_start);
    if (!surface || surface->Positions.empty()) {
        std::println("skipping {}: failed to load {}", name, obj_path.string());
        return {};
    }
    const auto num_verts = surface->Positions.size();
    std::vector<vec3> excite_positions(NumExcitePositions);
    for (uint32_t i = 0; i < NumExcitePositions; ++i) excite_positions[i] = surface->Positions[i * num_verts / NumExcitePositions];
    return LoadedSample{std::move(*surface), std::move(excite_positions), load_seconds};
}

std::unique_ptr<tetgenio> TryGenerateTets(const std::string &name, std::vector<vec3> positions, std::vector<uint32_t> triangle_indices, float ratio) {
    auto tets = GenerateTets(std::move(positions), std::move(triangle_indices), {.PreserveSurface = true, .SimplifyRatio = ratio});
    if (tets) return std::move(*tets);
    std::println("skipping {}: tetgen failed: {}", name, tets.error());
    return nullptr;
}

std::optional<BenchResult> RunSample(const fs::path &dataset, const std::string &name, float ratio, bool corpus) {
    auto sample = LoadSample(dataset, name);
    if (!sample) return {};
    auto &surface = sample->Surface;

    BenchResult r{.Name = name};
    const auto &material = MaterialForSample(name);
    r.Material = material.Name;
    r.LoadSeconds = sample->LoadSeconds;
    r.SurfaceTris = surface.TriangleIndices.size() / 3;

    try {
        const auto tets_start = std::chrono::steady_clock::now();
        const auto tets = TryGenerateTets(name, std::move(surface.Positions), std::move(surface.TriangleIndices), ratio);
        r.TetsSeconds = SecondsSince(tets_start);
        if (!tets) return {};
        r.TetCount = tets->numberoftetrahedra;

        const auto result = modal::mesh2modes(*tets, material.Properties, sample->ExcitePositions, vec3{1}, SolveConfig);
        r.Profile = result.Profile;
        r.NumModes = result.Modes.Freqs.size();
        r.FundamentalHz = r.NumModes > 0 ? result.Modes.Freqs.front() : 0;
        if (corpus) PrintCorpusSample(r, result);
        return r;
    } catch (const std::exception &e) {
        std::println("skipping {}: solve failed: {}", name, e.what());
    }
    return {};
}

// Synthetic interactive-edit loop. Solve once cold keeping the eigenpairs, then compare a cold
// re-solve against a reusing one for a Poisson ratio edit (warm-started subspace iteration) and
// a Young's modulus and density edit (analytically scaled eigenpairs, no eigensolve).
bool RunEditLoop(const fs::path &dataset, const std::string &name, float ratio) {
    auto sample = LoadSample(dataset, name);
    if (!sample) return false;

    const auto material = MaterialForSample(name).Properties;
    try {
        const auto tets = TryGenerateTets(name, std::move(sample->Surface.Positions), std::move(sample->Surface.TriangleIndices), ratio);
        if (!tets) return false;
        const auto initial = modal::mesh2modes(*tets, material, sample->ExcitePositions, vec3{1}, SolveConfig, {.KeepBasis = true});

        const auto f1 = [](const ModalModes &m) { return m.Freqs.empty() ? 0.f : m.Freqs.front(); };
        const auto timed_solve = [&](const AcousticMaterialProperties &edited, modal::SolveReuse reuse) {
            const auto start = std::chrono::steady_clock::now();
            auto result = modal::mesh2modes(*tets, edited, sample->ExcitePositions, vec3{1}, SolveConfig, reuse);
            return std::pair{std::move(result), SecondsSince(start)};
        };
        const auto report = [&](std::string_view label, const modal::ModalResult &cold, double cold_seconds, const ModalModes &reused, double reuse_seconds, uint32_t ops, uint32_t iters) {
            double max_gain_dev = 0;
            if (cold.Modes.Freqs.size() == reused.Freqs.size()) {
                for (size_t m = 0; m < cold.Modes.Freqs.size(); ++m) {
                    const double c = RmsGain(cold.Modes, m), w = RmsGain(reused, m);
                    if (c > 0) max_gain_dev = std::max(max_gain_dev, std::abs(w - c) / c);
                }
            }
            std::println(
                "{:<26} {:<11} cold {:>6.2f} s ({:>4} ops) -> reuse {:>5.2f} s ({:>4} ops, {:>2} iters): {:>7.2f}x | modes {} vs {}, f1 {:.1f} vs {:.1f}, max gain dev {:.1e}{}",
                name, label, cold_seconds, cold.Profile.OpApplications, reuse_seconds, ops, iters,
                cold_seconds / reuse_seconds, cold.Modes.Freqs.size(), reused.Freqs.size(), f1(cold.Modes), f1(reused), max_gain_dev,
                cold.Modes.Freqs.size() == reused.Freqs.size() && std::abs(f1(cold.Modes) - f1(reused)) < 0.05f ? "" : "  MISMATCH"
            );
        };

        auto nu_edited = material;
        nu_edited.PoissonRatio = std::min(nu_edited.PoissonRatio + 0.02, 0.49);
        {
            const auto [cold, cold_seconds] = timed_solve(nu_edited, {});
            const auto [warm, warm_seconds] = timed_solve(nu_edited, {.SeedBasis = &initial.Basis});
            report("nu edit:", cold, cold_seconds, warm.Modes, warm_seconds, warm.Profile.OpApplications, warm.Profile.Restarts);
        }

        auto scaled = material;
        scaled.YoungModulus *= 1.5;
        scaled.Density *= 0.8;
        {
            const auto [cold, cold_seconds] = timed_solve(scaled, {});
            const auto rescale_start = std::chrono::steady_clock::now();
            const auto rescaled = modal::RescaleModes(initial.Summary, initial.Modes, scaled, SolveConfig);
            const double rescale_seconds = SecondsSince(rescale_start);
            report("E/rho edit:", cold, cold_seconds, rescaled.value_or(ModalModes{}), rescale_seconds, 0, 0);
        }
        return true;
    } catch (const std::exception &e) {
        std::println("skipping {}: solve failed: {}", name, e.what());
    }
    return false;
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
    bool edit_loop = false;
    std::vector<std::string> names;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg{argv[i]};
        if (arg == "--all") all = true;
        else if (arg == "--corpus") corpus = true;
        else if (arg == "--edit-loop") edit_loop = true;
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

    if (edit_loop) {
        std::println("edit loop: perturb Poisson ratio by +0.02 at fixed tet topology, cold vs warm-started re-solve");
        bool any = false;
        for (const auto &name : names) any |= RunEditLoop(dataset, name, ratio);
        return any ? 0 : 1;
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
