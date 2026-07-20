#include "LoadObj.h"
#include "ValidateTetMesh.h"
#include "audio/AcousticMaterial.h"
#include "audio/RealImpact.h"
#include "audio/mesh2modes.h"
#include "mesh/Tets.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <mutex>
#include <print>
#include <pthread.h>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {
// A directory of .obj files under the corpus root, built by script/SetupTetCorpus.
struct Corpus {
    std::string_view Name;
    // Scale each model to this bounding box diagonal in metres, or zero to keep the file's own coordinates.
    float TargetDiagonal;
    // Material for a model the RealImpact mapping does not name.
    const AcousticMaterial *DefaultMaterial;
};

// RealImpact scans are in metres with a per-object material.
// Thingi10K models have neither, so they stand in as ceramic objects 0.30 m across, mid-range of RealImpact's 0.14 m to 0.48 m.
constexpr Corpus Corpora[]{
    {"realimpact", 0.f, &materials::acoustic::Ceramic},
    {"thingi10k", 0.30f, &materials::acoustic::Ceramic},
};
constexpr const Corpus &DefaultCorpus{Corpora[0]};

// A spread of sizes and materials. Any corpus model can be named on the command line instead.
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

// What a change is checked against.
// The corpora are not in the repo, so an unrecorded case is reported rather than assumed new.
const fs::path SnapshotPath{fs::path{MESHEDITOR_SOURCE_DIR} / "tests" / "fixtures" / "TetCorpusSnapshot.txt"};

// One case as a line: the mesh hash is the gate, and the counts say which phase moved when it trips.
struct CaseSignature {
    std::string Key;
    uint32_t Tets{}, Steiner{}, Flips{}, Splits{}, MissingEdges{}, MissingFaces{};
    uint64_t Hash{};
};

// FNV-1a over point bit patterns then tet indices, both in order, so the same tets reached a different way still differ.
uint64_t MeshHash(const TetMesh &mesh) {
    uint64_t h = 0xcbf29ce484222325ull;
    const auto mix = [&h](uint64_t v) {
        h ^= v;
        h *= 0x100000001b3ull;
    };
    for (const auto &p : mesh.Points) {
        mix(std::bit_cast<uint64_t>(p.x));
        mix(std::bit_cast<uint64_t>(p.y));
        mix(std::bit_cast<uint64_t>(p.z));
    }
    for (const auto &t : mesh.Tets) {
        for (const uint32_t v : t) mix(v);
    }
    return h;
}

std::string FormatSignature(const CaseSignature &sig) {
    return std::format("{:<34} {:>8} {:>6} {:>9} {:>6} {:>6} {:>6} {:016x}", sig.Key, sig.Tets, sig.Steiner, sig.Flips, sig.Splits, sig.MissingEdges, sig.MissingFaces, sig.Hash);
}

// Merges this run's cases into the file, since one run covers one corpus in one set of arms.
// Delete the file first to rebuild from nothing.
void WriteSnapshot(const std::vector<CaseSignature> &sigs) {
    std::map<std::string, std::string> lines;
    if (std::ifstream existing{SnapshotPath}) {
        for (std::string line; std::getline(existing, line);) {
            if (!line.empty() && line[0] != '#') lines.insert_or_assign(line.substr(0, line.find(' ')), line);
        }
    }
    for (const auto &sig : sigs) lines.insert_or_assign(sig.Key, FormatSignature(sig));

    std::ofstream file{SnapshotPath};
    file << "# Every corpus case as the tetrahedralizer produces it, written by --snapshot write and compared by --snapshot check.\n"
            "# The mesh hash is the gate, and the counts say which phase moved when it trips.\n"
            "# The corpora are not in the repo, so a missing case means a mismatched corpus, not a regression.\n"
            "# case                                  tets steiner     flips splits missE missF mesh-hash\n";
    for (const auto &[key, line] : lines) file << line << '\n';
    std::println("snapshot: wrote {} cases, {} in the file", sigs.size(), lines.size());
}

// Reports every case whose line moved, and every recorded case this run should have covered and did not.
// Cases outside this run's corpus and arms are left alone.
int CheckSnapshot(const std::vector<CaseSignature> &sigs) {
    std::map<std::string, std::string> recorded;
    std::ifstream file{SnapshotPath};
    if (!file) {
        std::println("snapshot: {} is missing, run --snapshot write", SnapshotPath.string());
        return 1;
    }
    for (std::string line; std::getline(file, line);) {
        if (line.empty() || line[0] == '#') continue;
        recorded.emplace(line.substr(0, line.find(' ')), line);
    }
    std::set<std::string> corpora, arms;
    for (const auto &sig : sigs) {
        corpora.insert(sig.Key.substr(0, sig.Key.find('/')));
        arms.insert(sig.Key.substr(sig.Key.rfind('/')));
    }

    int fails = 0;
    for (const auto &sig : sigs) {
        const auto it = recorded.find(sig.Key);
        if (it == recorded.end()) {
            std::println("snapshot: {} is not in the file", sig.Key);
            ++fails;
            continue;
        }
        if (const auto line = FormatSignature(sig); line != it->second) {
            std::println("snapshot: {} changed\n    was {}\n    now {}", sig.Key, it->second, line);
            ++fails;
        }
        recorded.erase(it);
    }
    for (const auto &[key, line] : recorded) {
        if (!corpora.contains(key.substr(0, key.find('/'))) || !arms.contains(key.substr(key.rfind('/')))) continue;
        std::println("snapshot: {} did not run", key);
        ++fails;
    }
    std::println("snapshot: {} cases, {} differences", sigs.size(), fails);
    return fails;
}

struct BenchResult {
    std::string Name, Material;
    size_t SurfaceTris{}, TetCount{};
    modal::SolveProfile Profile;
    double LoadSeconds{}, TetsSeconds{};
    size_t NumModes{};
    double FundamentalHz{};

    double Total() const {
        const auto &p = Profile;
        return LoadSeconds + TetsSeconds + p.MassProps + p.QuadMesh + p.Assemble + p.SampleExcite + p.Factorize + p.Iterate + p.Extract;
    }
};

double SecondsSince(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

template<class... A> void Line(std::string &out, std::format_string<A...> fmt, A &&...args) {
    out += std::format(fmt, std::forward<A>(args)...);
    out += '\n';
}

const AcousticMaterial &MaterialForSample(const Corpus &corpus, const std::string &name) {
    const auto object_name = RealImpact::FindObjectName(fs::path{name});
    const auto material_name = object_name ? RealImpact::FindMaterialName(*object_name) : std::nullopt;
    const auto *material = material_name ? materials::acoustic::Find(*material_name) : nullptr;
    return material ? *material : *corpus.DefaultMaterial;
}

// RMS shape magnitude of one mode across excitation positions (mode shape signs are arbitrary).
double RmsGain(const ModalModes &modes, size_t m) {
    double gain_sq = 0;
    for (const auto &shape : modes.Shapes) gain_sq += double(glm::dot(shape[m], shape[m]));
    return std::sqrt(gain_sq / modes.Shapes.size());
}

// A model's surface, excitation positions like the app's default, and the factor scaling the result to the corpus's physical size.
struct LoadedSample {
    SurfaceMesh Surface;
    std::vector<vec3> ExcitePositions;
    double LoadSeconds{};
    double SolveScale{1};
};

std::optional<LoadedSample> LoadSample(const Corpus &corpus, const fs::path &obj_path, std::string &out) {
    const auto load_start = std::chrono::steady_clock::now();
    auto surface = LoadObj(obj_path);
    const double load_seconds = SecondsSince(load_start);
    if (!surface || surface->Positions.empty()) {
        Line(out, "skipping {}: failed to load", obj_path.string());
        return {};
    }
    double scale = 1;
    if (corpus.TargetDiagonal > 0) {
        vec3 lo = surface->Positions.front(), hi = lo;
        for (const auto &p : surface->Positions) {
            lo = glm::min(lo, p);
            hi = glm::max(hi, p);
        }
        const double diagonal = glm::length(dvec3{hi - lo});
        scale = diagonal > 0 ? corpus.TargetDiagonal / diagonal : 1;
    }
    const auto num_verts = surface->Positions.size();
    std::vector<vec3> excite_positions(NumExcitePositions);
    for (uint32_t i = 0; i < NumExcitePositions; ++i) excite_positions[i] = surface->Positions[i * num_verts / NumExcitePositions];
    return LoadedSample{std::move(*surface), std::move(excite_positions), load_seconds, scale};
}

// Every tetrahedralizer tolerance derives from the input bounding box, so models are tetrahedralized in their own coordinates and only the result is scaled.
void ScaleToMetres(TetMesh &mesh, std::vector<vec3> &excite_positions, double scale) {
    if (scale == 1) return;
    for (auto &p : mesh.Points) p *= scale;
    for (auto &p : excite_positions) p *= float(scale);
}

// One object across every arm.
struct ObjectResult {
    std::string Out;
    std::vector<CaseSignature> Sigs;
    std::optional<BenchResult> Bench;
    int Fails{};
};

struct RunOptions {
    float Ratio{1};
    bool Tets{true}, Modes{true};
    // One block per object rather than one timing row.
    bool Blocks{false};
    // Run the quality arm beside the base one, as the snapshot records both.
    bool QualityArm{false};
};

ObjectResult RunObject(const Corpus &corpus, const fs::path &obj_path, const RunOptions &opts) {
    ObjectResult r;
    const auto name = obj_path.stem().string();
    auto sample = LoadSample(corpus, obj_path, r.Out);
    if (!sample) {
        r.Fails = 1;
        return r;
    }

    const auto &material = MaterialForSample(corpus, name);
    BenchResult bench{
        .Name = name,
        .Material = std::string{material.Name},
        .LoadSeconds = sample->LoadSeconds,
    };
    if (opts.Blocks) Line(r.Out, "=== {} ({}) ===", name, material.Name);

    if (!opts.Tets) {
        bench.SurfaceTris = sample->Surface.TriangleIndices.size() / 3;
        Line(r.Out, "{:<28} {:>6} pts {:>6} tris", name, sample->Surface.Positions.size(), bench.SurfaceTris);
        r.Bench = std::move(bench);
        return r;
    }

    // Kept so every arm builds from the same input and the result can be validated against it.
    auto positions = std::move(sample->Surface.Positions);
    auto tris = std::move(sample->Surface.TriangleIndices);
    SimplifySurface(positions, tris, opts.Ratio);
    const std::vector<dvec3> points(positions.begin(), positions.end());
    bench.SurfaceTris = tris.size() / 3;

    for (int arm = 0; arm <= int(opts.QualityArm); ++arm) {
        const tetra::Options arm_opts{.Quality = arm > 0};
        const auto tets_start = std::chrono::steady_clock::now();
        auto tets = GenerateTets(positions, tris, arm_opts);
        const double tets_seconds = SecondsSince(tets_start);
        if (!tets) {
            Line(r.Out, "{:<28} FAILED: {}", name, tets.error());
            ++r.Fails;
            continue;
        }
        const auto &p = tets->Profile;
        const auto error = ValidateTetMesh(points, tris, tets->Mesh);
        if (!error.empty()) ++r.Fails;

        r.Sigs.push_back({
            .Key = std::format("{}/{}@{:.2f}/{}", corpus.Name, name, opts.Ratio, arm_opts.Quality ? "q" : "noq"),
            .Tets = p.TetCount,
            .Steiner = p.SteinerCount,
            .Flips = p.FlipCount,
            .Splits = p.SplitCount,
            .MissingEdges = uint32_t(p.MissingEdgeCount),
            .MissingFaces = uint32_t(p.MissingFaceCount),
            .Hash = MeshHash(tets->Mesh),
        });
        // The timing table carries these sizes in its own row, so it only wants what went wrong.
        if (opts.Blocks || !opts.Modes || !error.empty()) {
            Line(r.Out, "{:<28} {:>6} pts {:>6} tris -> {:>7} tets, {:>5} steiner | {:>8.5f} s | flips {:>6} splits {:>4} missE {:>4} missF {:>4} | {}", name + (arm_opts.Quality ? " q" : ""), points.size(), tris.size() / 3, p.TetCount, p.SteinerCount, tets_seconds, p.FlipCount, p.SplitCount, p.MissingEdgeCount, p.MissingFaceCount, error.empty() ? "OK" : ("INVALID: " + error));
        }

        // Modes come from the base arm, the tetrahedralization the app produces.
        if (arm == 0) {
            bench.TetsSeconds = tets_seconds;
            bench.TetCount = tets->Mesh.Tets.size();
        }
        if (arm > 0 || !opts.Modes) continue;

        ScaleToMetres(tets->Mesh, sample->ExcitePositions, sample->SolveScale);
        try {
            const auto result = modal::mesh2modes(tets->Mesh, material.Properties, sample->ExcitePositions, vec3{1}, SolveConfig);
            const auto &sp = result.Profile;
            bench.Profile = sp;
            bench.NumModes = result.Modes.Freqs.size();
            bench.FundamentalHz = bench.NumModes > 0 ? result.Modes.Freqs.front() : 0;
            if (!opts.Blocks) continue;

            Line(r.Out, "geometry: {} tris -> {} tets, {} dofs, {} nnz(K)", bench.SurfaceTris, bench.TetCount, sp.Dofs, sp.StiffnessNonZeros);
            Line(r.Out, "profile: tets {:.2f} s, assemble {:.2f} s, factorize {:.2f} s, iterate {:.2f} s (opsolve {:.2f} s), {} ops, {} restarts", bench.TetsSeconds, sp.Assemble, sp.Factorize, sp.Iterate, sp.OpSolve, sp.OpApplications, sp.Restarts);
            Line(r.Out, "tets: delaunay {:.3f} s, recover {:.3f} s, carve {:.3f} s, refine {:.3f} s, {} flips, {} splits, {} missing edges, {} missing faces, {} steiner", p.DelaunaySeconds, p.RecoverSeconds, p.CarveSeconds, p.RefineSeconds, p.FlipCount, p.SplitCount, p.MissingEdgeCount, p.MissingFaceCount, p.SteinerCount);
            Line(r.Out, "mass: {:.4g} kg", result.MassProps.Mass);
            Line(r.Out, "modes: {}", result.Modes.Freqs.size());
            for (size_t m = 0; m < result.Modes.Freqs.size(); ++m) {
                Line(r.Out, "  {:>2}: {:>9.2f} Hz, t60 {:>7.3f} s, gain {:.3e}", m + 1, result.Modes.Freqs[m], result.Modes.T60s[m], RmsGain(result.Modes, m));
            }
            Line(r.Out, "");
        } catch (const std::exception &e) {
            Line(r.Out, "skipping {}: solve failed: {}", name, e.what());
            ++r.Fails;
        }
    }
    r.Bench = std::move(bench);
    return r;
}

// Synthetic interactive-edit loop. Solve once cold keeping the eigenpairs, then compare a cold
// re-solve against a reusing one for a Poisson ratio edit (warm-started subspace iteration) and
// a Young's modulus and density edit (analytically scaled eigenpairs, no eigensolve).
bool RunEditLoop(const Corpus &corpus, const fs::path &obj_path, float ratio) {
    std::string out;
    const auto name = obj_path.stem().string();
    auto sample = LoadSample(corpus, obj_path, out);
    std::fputs(out.c_str(), stdout);
    if (!sample) return false;

    const auto material = MaterialForSample(corpus, name).Properties;
    try {
        auto positions = std::move(sample->Surface.Positions);
        auto tris = std::move(sample->Surface.TriangleIndices);
        SimplifySurface(positions, tris, ratio);
        auto tets = GenerateTets(std::move(positions), std::move(tris), {});
        if (!tets) {
            std::println("skipping {}: tetrahedralization failed: {}", name, tets.error());
            return false;
        }
        ScaleToMetres(tets->Mesh, sample->ExcitePositions, sample->SolveScale);
        const auto initial = modal::mesh2modes(tets->Mesh, material, sample->ExcitePositions, vec3{1}, SolveConfig, {.KeepBasis = true});

        const auto f1 = [](const ModalModes &m) { return m.Freqs.empty() ? 0.f : m.Freqs.front(); };
        const auto timed_solve = [&](const AcousticMaterialProperties &edited, modal::SolveReuse reuse) {
            const auto start = std::chrono::steady_clock::now();
            auto result = modal::mesh2modes(tets->Mesh, edited, sample->ExcitePositions, vec3{1}, SolveConfig, reuse);
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
    std::println("{:<26} {:>8} {:>8} {:>8} {:>8} {:>11} | {:>6} {:>7} {:>6} {:>6} {:>7} {:>8} {:>8} {:>8} {:>6} {:>6} {:>8} | {:>5} {:>5} {:>4} {:>9}", "sample", "material", "tris", "tets", "dofs", "nnz(K)", "load", "tets", "mass", "quad", "asm", "factor", "iter", "opsolve", "excite", "extr", "total", "modes", "ops", "rst", "f1");
}

void PrintRow(const BenchResult &r) {
    const auto &p = r.Profile;
    std::println("{:<26} {:>8} {:>8} {:>8} {:>8} {:>11} | {:>6.2f} {:>7.2f} {:>6.2f} {:>6.2f} {:>7.2f} {:>8.2f} {:>8.2f} {:>8.2f} {:>6.2f} {:>6.2f} {:>8.2f} | {:>5} {:>5} {:>4} {:>9.1f}", r.Name, r.Material, r.SurfaceTris, r.TetCount, p.Dofs, p.StiffnessNonZeros, r.LoadSeconds, r.TetsSeconds, p.MassProps, p.QuadMesh, p.Assemble, p.Factorize, p.Iterate, p.OpSolve, p.SampleExcite, p.Extract, r.Total(), r.NumModes, p.OpApplications, p.Restarts, r.FundamentalHz);
}

void PrintSummary(const std::vector<BenchResult> &results, float ratio) {
    double load = 0, tets = 0, grand = 0;
    modal::SolveProfile sum;
    for (const auto &r : results) {
        load += r.LoadSeconds;
        tets += r.TetsSeconds;
        sum += r.Profile;
        grand += r.Total();
    }
    if (grand <= 0) return;

    std::println("\n--- totals across {} samples, simplify ratio {:.2f}: {:.2f} s ---", results.size(), ratio, grand);
    const auto line = [&](std::string_view section, double seconds) {
        std::println("{:>10}: {:>8.2f} s {:>5.1f}%", section, seconds, 100 * seconds / grand);
    };
    line("load", load);
    line("tets", tets);
    if (sum.Assemble <= 0) return;

    line("mass", sum.MassProps);
    line("quad", sum.QuadMesh);
    line("assemble", sum.Assemble);
    line("factorize", sum.Factorize);
    line("iterate", sum.Iterate);
    std::println("{:>10}: {:>8.2f} s ({:.1f}% of iterate)", "opsolve", sum.OpSolve, sum.Iterate > 0 ? 100 * sum.OpSolve / sum.Iterate : 0);
    line("excite", sum.SampleExcite);
    line("extract", sum.Extract);
}

// Corpus models, ordered by the number their names start with.
std::vector<fs::path> CorpusObjects(const fs::path &dir, const std::vector<std::string> &names) {
    std::vector<fs::path> objs;
    for (const auto &entry : fs::directory_iterator{dir}) {
        if (entry.path().extension() != ".obj") continue;
        const auto stem = entry.path().stem().string();
        if (!names.empty() && std::ranges::find(names, stem) == names.end()) continue;
        objs.push_back(entry.path());
    }
    std::ranges::sort(objs, {}, [](const fs::path &p) {
        const auto stem = p.stem().string();
        return std::pair{std::atoi(stem.c_str()), stem};
    });
    return objs;
}

// Objects print in corpus order as they become ready, so a stalled one still names everything before it.
// Recovery recurses past the 512 KB a std::thread gets, so workers are raw pthreads with room to spare.
std::vector<ObjectResult> RunObjects(const Corpus &corpus, const std::vector<fs::path> &objs, const RunOptions &opts, unsigned threads) {
    std::vector<ObjectResult> results(objs.size());
    std::vector<char> done(objs.size(), 0);
    std::atomic<size_t> next{0};
    std::mutex mutex;
    std::condition_variable ready;
    // Biggest first, so the run ends on small objects rather than waiting on a late large one.
    std::vector<size_t> order(objs.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::ranges::sort(order, std::ranges::greater{}, [&](size_t i) { return fs::file_size(objs[i]); });

    const auto work = [&] {
        for (size_t k = next++; k < objs.size(); k = next++) {
            const size_t i = order[k];
            results[i] = RunObject(corpus, objs[i], opts);
            {
                std::lock_guard lock{mutex};
                done[i] = 1;
            }
            ready.notify_all();
        }
    };
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 512 * 1024 * 1024);
    std::vector<pthread_t> pool(threads);
    for (auto &thread : pool) {
        pthread_create(
            &thread, &attr,
            [](void *arg) -> void * {
                (*static_cast<const decltype(work) *>(arg))();
                return nullptr;
            },
            const_cast<void *>(static_cast<const void *>(&work))
        );
    }
    pthread_attr_destroy(&attr);
    for (size_t i = 0; i < objs.size(); ++i) {
        std::unique_lock lock{mutex};
        ready.wait(lock, [&, i] { return done[i] != 0; });
        lock.unlock();
        std::fputs(results[i].Out.c_str(), stdout);
        results[i].Out.clear();
        results[i].Out.shrink_to_fit();
    }
    for (auto &thread : pool) pthread_join(thread, nullptr);
    return results;
}

const Corpus *FindCorpus(std::string_view name) {
    for (const auto &c : Corpora) {
        if (c.Name == name) return &c;
    }
    return nullptr;
}
} // namespace

int main(int argc, char **argv) {
    // Line buffering so a redirected run shows every object that finished before a hang.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    fs::path root{TET_CORPUS_DIR};
    RunOptions opts;
    bool edit_loop = false, explicit_modes = false;
    unsigned jobs = 0;
    std::string_view snapshot;
    std::vector<std::string> corpus_names, names;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg{argv[i]};
        if (arg == "--corpus") opts.Blocks = true;
        else if (arg == "--edit-loop") edit_loop = true;
        else if (arg == "--tets") opts.Tets = true;
        else if (arg == "--no-tets") opts.Tets = false;
        else if (arg == "--modes") opts.Modes = explicit_modes = true;
        else if (arg == "--no-modes") opts.Modes = false;
        else if (arg == "--ratio" && i + 1 < argc) opts.Ratio = std::stof(argv[++i]);
        else if (arg == "--jobs" && i + 1 < argc) jobs = unsigned(std::stoul(argv[++i]));
        else if (arg == "--snapshot" && i + 1 < argc) snapshot = argv[++i];
        else if (arg == "--corpus-dir" && i + 1 < argc) root = argv[++i];
        else if (arg == "--dataset" && i + 1 < argc) corpus_names.emplace_back(argv[++i]);
        else names.emplace_back(arg);
    }
    const bool snap_write = snapshot == "write", snap_check = snapshot == "check";
    if (!snapshot.empty() && !snap_write && !snap_check) {
        std::println("--snapshot takes 'write' or 'check'");
        return 1;
    }
    // A signature holds nothing the solve produces, so a snapshot run skips modes unless asked, and records both quality arms.
    if (snap_write || snap_check) {
        opts.Tets = opts.QualityArm = true;
        opts.Modes = explicit_modes;
    }
    // The tetrahedralization feeds the solve, so there are no modes without it.
    if (!opts.Tets) opts.Modes = false;
    // The named spread is the timing table's default, so it applies only when no corpus was named.
    if (names.empty() && corpus_names.empty() && !opts.Blocks && !snap_write && !snap_check) {
        names.assign(std::begin(DefaultSamples), std::end(DefaultSamples));
    }
    if (corpus_names.empty()) corpus_names.emplace_back(DefaultCorpus.Name);
    if (!fs::is_directory(root)) {
        std::println("corpus root not found: {} (run script/SetupTetCorpus)", root.string());
        return 1;
    }
    // Each build carries its own random state, so an object reaches the same mesh whatever runs beside it.
    // A solve is memory-heavy and its timings are what the corpus output tracks, so modes run one at a time.
    const unsigned threads = jobs > 0 ? jobs : (opts.Modes ? 1u : std::max(1u, std::thread::hardware_concurrency()));

    if (opts.Blocks) {
        std::println("modal solve corpus: {}, simplify ratio {:.2f}, {} excitation positions", corpus_names.size() == 1 ? corpus_names[0] : "multiple corpora", opts.Ratio, NumExcitePositions);
        std::println("solver config: modes {:g}-{:g} Hz, keep {} of {} eigenpairs, tolerance {:g}, max restarts {}", SolveConfig.MinModeFreq, SolveConfig.MaxModeFreq, SolveConfig.NumModes, SolveConfig.NumFemModes, SolveConfig.Tolerance, SolveConfig.MaxRestarts);
        std::println("");
    } else if (opts.Modes) {
        PrintHeader();
    }

    int fails = 0;
    std::vector<BenchResult> bench_results;
    std::vector<CaseSignature> sigs;
    for (const auto &corpus_name : corpus_names) {
        const auto *corpus = FindCorpus(corpus_name);
        if (corpus == nullptr) {
            std::println("unknown corpus '{}'", corpus_name);
            return 1;
        }
        const auto dir = root / corpus_name;
        if (!fs::is_directory(dir)) {
            std::println("corpus directory not found: {} (run script/SetupTetCorpus)", dir.string());
            return 1;
        }
        const auto objs = CorpusObjects(dir, names);
        if (objs.empty()) {
            std::println("no models in {}", dir.string());
            return 1;
        }
        if (edit_loop) {
            std::println("edit loop: perturb Poisson ratio by +0.02 at fixed tet topology, cold vs warm-started re-solve");
            bool any = false;
            for (const auto &obj : objs) any |= RunEditLoop(*corpus, obj, opts.Ratio);
            if (!any) return 1;
            continue;
        }
        if (threads > 1) std::println("corpus {}: {} objects, {} at a time", corpus_name, objs.size(), threads);
        for (auto &r : RunObjects(*corpus, objs, opts, threads)) {
            fails += r.Fails;
            sigs.insert(sigs.end(), std::make_move_iterator(r.Sigs.begin()), std::make_move_iterator(r.Sigs.end()));
            if (!r.Bench) continue;
            if (!opts.Blocks && opts.Modes) PrintRow(*r.Bench);
            bench_results.push_back(std::move(*r.Bench));
        }
    }
    if (edit_loop) return 0;

    PrintSummary(bench_results, opts.Ratio);
    if (snap_write || snap_check) {
        std::ranges::sort(sigs, {}, &CaseSignature::Key);
        if (snap_write) WriteSnapshot(sigs);
        else fails += CheckSnapshot(sigs);
    }
    if (fails > 0) std::println("{} failures", fails);
    return bench_results.empty() || fails > 0 ? 1 : 0;
}
