#include "LoadObj.h"
#include "RunSuites.h"
#include "ValidateTetMesh.h"
#include "audio/AcousticMaterialProperties.h"
#include "audio/mesh2modes.h"
#include "mesh/Tets.h"

#include <boost/ut.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <numbers>
#include <print>
#include <random>
#include <vector>

using namespace boost::ut;

namespace {
// Free-free rectangular prism whose mode families have closed forms (with Poisson's ratio 0):
//   Longitudinal: f_n = n*sqrt(E/rho)/(2L)
//   Torsional (square section): f_n = n*sqrt(G*J/(rho*Ip))/(2L), G = E/2, J = 0.140577*a^4, Ip = a^4/6
//   Bending (Euler-Bernoulli): f_i = (bL)_i^2/(2*pi) * sqrt(E/rho)*r_g/L^2, (bL) = {4.73004, 7.85320, 10.99561},
//     with r_g the section's radius of gyration about the bending axis (thickness/sqrt(12)).
struct Bar {
    double Length, Width, Thickness; // x, y, z extents, meters
    AcousticMaterialProperties Material;
};

constexpr double BendingBL[]{4.73004074, 7.85320462, 10.9956078};

// Structured tet mesh of the bar: (nx+1)*(ny+1)*(nz+1) vertices, each grid cell split
// into six tetrahedra around its main diagonal (Kuhn subdivision).
TetMesh MakeBarTets(const Bar &bar, int nx, int ny, int nz) {
    TetMesh tets;
    const int vx = nx + 1, vy = ny + 1, vz = nz + 1;
    tets.Points.resize(vx * vy * vz);
    const auto id = [&](int i, int j, int k) { return uint32_t((i * vy + j) * vz + k); };
    for (int i = 0; i < vx; ++i) {
        for (int j = 0; j < vy; ++j) {
            for (int k = 0; k < vz; ++k) {
                tets.Points[id(i, j, k)] = {bar.Length * i / nx, bar.Width * j / ny, bar.Thickness * k / nz};
            }
        }
    }
    tets.Tets.reserve(nx * ny * nz * 6);
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            for (int k = 0; k < nz; ++k) {
                const uint32_t c[8]{
                    id(i, j, k), id(i + 1, j, k), id(i, j + 1, k), id(i + 1, j + 1, k),
                    id(i, j, k + 1), id(i + 1, j, k + 1), id(i, j + 1, k + 1), id(i + 1, j + 1, k + 1)
                };
                // Six tets sharing the c[0]-c[7] diagonal, one per axis-order path between them.
                static constexpr int Corners[6][4]{
                    {0, 1, 3, 7}, {0, 3, 2, 7}, {0, 2, 6, 7}, {0, 6, 4, 7}, {0, 4, 5, 7}, {0, 5, 1, 7}
                };
                for (const auto &tet : Corners) {
                    tets.Tets.push_back({c[tet[0]], c[tet[1]], c[tet[2]], c[tet[3]]});
                }
            }
        }
    }
    return tets;
}

enum class Family {
    Longitudinal,
    Torsional,
    Bending, // Lateral translation in either plane (square-section pairs are degenerate and may mix planes)
    BendingY, // Lateral translation dominantly along y
    BendingZ, // Lateral translation dominantly along z
    Other,
};

// Classify a mode by its shape's kinetic-energy fractions. Torsion is measured as the energy of the
// best-fit rigid rotation of each cross-section slice about the bar axis, so lateral translation
// (which also moves tangentially relative to the axis) does not read as torsion.
Family Classify(const ModalModes &modes, uint32_t mode, const Bar &bar, int nx) {
    double axial = 0, lateral_y = 0, lateral_z = 0, total = 0;
    std::map<int, std::pair<double, double>> slices; // x index -> (sum of r x u along the axis, sum of r^2)
    for (size_t v = 0; v < modes.Positions.size(); ++v) {
        const vec3 u = modes.Shapes[v][mode];
        const vec3 p = modes.Positions[v];
        const double ry = p.y - bar.Width / 2, rz = p.z - bar.Thickness / 2;
        axial += double(u.x) * u.x;
        lateral_y += double(u.y) * u.y;
        lateral_z += double(u.z) * u.z;
        total += glm::dot(u, u);
        auto &[circulation, r2] = slices[int(std::lround(p.x * nx / bar.Length))];
        circulation += ry * u.z - rz * u.y;
        r2 += ry * ry + rz * rz;
    }
    if (total <= 0) return Family::Other;

    double rotation = 0; // Energy of the per-slice fitted rigid rotation
    for (const auto &[_, slice] : slices) {
        const auto &[circulation, r2] = slice;
        if (r2 > 0) rotation += circulation * circulation / r2;
    }
    if (axial / total > 0.85) return Family::Longitudinal;
    if (rotation / total > 0.85) return Family::Torsional;
    // Bending carries some axial rotary motion for stubby sections, so the lateral threshold stays loose.
    if (const double lateral = lateral_y + lateral_z; lateral / total > 0.6 && rotation / total < 0.5) {
        if (lateral_y / lateral > 0.8) return Family::BendingY;
        if (lateral_z / lateral > 0.8) return Family::BendingZ;
        return Family::Bending;
    }
    return Family::Other;
}

// Solve the bar and bucket FEM frequencies by classified family, ascending.
std::map<Family, std::vector<double>> SolveBar(const Bar &bar, int nx, int ny, int nz) {
    const auto tets = MakeBarTets(bar, nx, ny, nz);
    const std::vector<vec3> all_positions(tets.Points.begin(), tets.Points.end());
    const auto result = modal::mesh2modes(tets, bar.Material, all_positions, vec3{1}, {});
    std::map<Family, std::vector<double>> fem;
    for (uint32_t mode = 0; mode < result.Modes.Freqs.size(); ++mode) {
        fem[Classify(result.Modes, mode, bar, nx)].push_back(result.Modes.Freqs[mode]);
    }
    return fem;
}

std::vector<double> HarmonicSeries(double f1) { return {f1, 2 * f1, 3 * f1}; }

std::vector<double> BendingTheory(const Bar &bar, double thickness, int per_root) {
    const double r_gyration = thickness / std::sqrt(12.0);
    const double base = std::sqrt(bar.Material.YoungModulus / bar.Material.Density) * r_gyration / (2 * std::numbers::pi * bar.Length * bar.Length);
    std::vector<double> freqs;
    for (const double bl : BendingBL) {
        for (int i = 0; i < per_root; ++i) freqs.push_back(bl * bl * base);
    }
    return freqs;
}

// A closed triangle surface in the tetrahedralizer's double coordinates.
struct Surface {
    std::vector<dvec3> Points;
    std::vector<uint32_t> Tris;
};

// Axis-aligned box as a k x k grid per face, heavy in exact degeneracies.
// k of 1 is the unit cube.
Surface GridBox(int k) {
    Surface s;
    std::map<std::array<int, 3>, uint32_t> ids;
    const auto vid = [&](int x, int y, int z) {
        const auto [it, inserted] = ids.try_emplace(std::array{x, y, z}, uint32_t(s.Points.size()));
        if (inserted) s.Points.emplace_back(double(x) / k, double(y) / k, double(z) / k);
        return it->second;
    };
    const auto face = [&](auto &&corner) {
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < k; ++j) {
                const uint32_t a = corner(i, j), b = corner(i + 1, j), c = corner(i + 1, j + 1), d = corner(i, j + 1);
                s.Tris.insert(s.Tris.end(), {a, b, c, a, c, d});
            }
        }
    };
    face([&](int i, int j) { return vid(i, j, 0); });
    face([&](int i, int j) { return vid(i, j, k); });
    face([&](int i, int j) { return vid(i, 0, j); });
    face([&](int i, int j) { return vid(i, k, j); });
    face([&](int i, int j) { return vid(0, i, j); });
    face([&](int i, int j) { return vid(k, i, j); });
    return s;
}

// Icosphere with optional radial noise.
Surface Sphere(int subdivisions, double noise, unsigned seed) {
    constexpr double phi = std::numbers::phi;
    std::vector<dvec3> pts{{-1, phi, 0}, {1, phi, 0}, {-1, -phi, 0}, {1, -phi, 0}, {0, -1, phi}, {0, 1, phi}, {0, -1, -phi}, {0, 1, -phi}, {phi, 0, -1}, {phi, 0, 1}, {-phi, 0, -1}, {-phi, 0, 1}};
    std::vector<std::array<uint32_t, 3>> tris{
        {0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11}, {1, 5, 9}, {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8}, {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8}, {3, 8, 9}, {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}
    };
    for (auto &p : pts) p = glm::normalize(p);
    for (int s = 0; s < subdivisions; ++s) {
        std::map<uint64_t, uint32_t> mid;
        const auto midpoint = [&](uint32_t a, uint32_t b) {
            const uint64_t key = (uint64_t(std::min(a, b)) << 32) | std::max(a, b);
            const auto [it, inserted] = mid.try_emplace(key, uint32_t(pts.size()));
            if (inserted) pts.push_back(glm::normalize(0.5 * (pts[a] + pts[b])));
            return it->second;
        };
        std::vector<std::array<uint32_t, 3>> next;
        for (const auto &t : tris) {
            const uint32_t ab = midpoint(t[0], t[1]), bc = midpoint(t[1], t[2]), ca = midpoint(t[2], t[0]);
            next.insert(next.end(), {{t[0], ab, ca}, {t[1], bc, ab}, {t[2], ca, bc}, {ab, bc, ca}});
        }
        tris = std::move(next);
    }
    if (noise > 0) {
        std::mt19937 rng{seed};
        std::uniform_real_distribution<double> d{1 - noise, 1 + noise};
        for (auto &p : pts) p *= d(rng);
    }
    Surface s;
    s.Points = std::move(pts);
    for (const auto &t : tris) s.Tris.insert(s.Tris.end(), {t[0], t[1], t[2]});
    return s;
}

// The RealImpact dataset, which REALIMPACT_DATASET_DIR overrides at run time.
std::filesystem::path DatasetDir() {
    const char *env_dataset = std::getenv("REALIMPACT_DATASET_DIR");
    return env_dataset ? env_dataset : REALIMPACT_DATASET_DIR;
}

// Compares the lowest computed modes of a family against theory.
void CheckFamily(std::string_view name, const std::vector<double> &fem, const std::vector<double> &theory, double tolerance, size_t min_count = 2) {
    const auto count = std::min(fem.size(), theory.size());
    expect(count >= min_count);
    for (size_t i = 0; i < count; ++i) {
        const double ratio = fem[i] / theory[i];
        std::println("{:>12} {}: theory {:8.2f} Hz, FEM {:8.2f} Hz, ratio {:.4f}", name, i + 1, theory[i], fem[i], ratio);
        expect(std::abs(ratio - 1.0) < tolerance);
    }
}
} // namespace

int main() {
    // Square section: longitudinal validates the E/rho/assembly/eigensolve chain end to end,
    // torsion and bending validate shear response.
    "square bar modes match closed forms"_test = [] {
        const Bar bar{.Length = 0.3, .Width = 0.05, .Thickness = 0.05, .Material = {.Density = 1000, .YoungModulus = 1e7, .PoissonRatio = 0, .Alpha = 0, .Beta = 0}};
        const double speed = std::sqrt(bar.Material.YoungModulus / bar.Material.Density);
        constexpr double TorsionOverPolar = 0.140577 * 6; // J/Ip for a square section
        const double torsion_f1 = std::sqrt(bar.Material.Mu() / bar.Material.Density * TorsionOverPolar) / (2 * bar.Length);
        std::println("--- square bar 20x4x4 ---");
        auto fem = SolveBar(bar, 20, 4, 4);
        // Degenerate pairs may mix planes, so pool all bending buckets.
        auto bending = fem[Family::Bending];
        for (const auto f : {Family::BendingY, Family::BendingZ}) bending.insert(bending.end(), fem[f].begin(), fem[f].end());
        std::ranges::sort(bending);
        CheckFamily("longitudinal", fem[Family::Longitudinal], HarmonicSeries(speed / (2 * bar.Length)), 0.01);
        CheckFamily("torsional", fem[Family::Torsional], HarmonicSeries(torsion_f1), 0.05);
        // Euler-Bernoulli overestimates higher modes of a stubby bar (no shear/rotary inertia), so
        // compare only the first degenerate pair.
        bending.resize(std::min<size_t>(bending.size(), 2));
        CheckFamily("bending", bending, BendingTheory(bar, bar.Thickness, 2), 0.10);
    };

    // Thin section with a single element through the thickness, matching how thin-walled objects
    // tetrahedralize in practice. Quadratic elements capture the bending strain through one element.
    "thin bar bending matches closed forms"_test = [] {
        const Bar bar{.Length = 0.3, .Width = 0.05, .Thickness = 0.01, .Material = {.Density = 1000, .YoungModulus = 1e9, .PoissonRatio = 0, .Alpha = 0, .Beta = 0}};
        const double speed = std::sqrt(bar.Material.YoungModulus / bar.Material.Density);
        std::println("--- thin bar 30x5x1 ---");
        auto fem = SolveBar(bar, 30, 5, 1);
        CheckFamily("longitudinal", fem[Family::Longitudinal], HarmonicSeries(speed / (2 * bar.Length)), 0.01);
        // Euler-Bernoulli overestimates the stiff plane's higher modes (no shear/rotary inertia), so
        // compare only its first mode.
        auto bending_y_theory = BendingTheory(bar, bar.Width, 1);
        bending_y_theory.resize(1);
        CheckFamily("bending-y", fem[Family::BendingY], bending_y_theory, 0.10, 1);
        CheckFamily("bending-z", fem[Family::BendingZ], BendingTheory(bar, bar.Thickness, 1), 0.05);
    };

    // Exact vertex coordinates put the predicates on their degenerate cases: coplanar faces, cospherical corners, collinear edges.
    // Noise on the sphere moves them to near-degenerate instead.
    "synthetic shapes tetrahedralize to valid meshes"_test = [] {
        const std::pair<std::string_view, Surface> cases[]{
            {"cube", GridBox(1)},
            {"grid box 4", GridBox(4)},
            {"grid box 7", GridBox(7)},
            {"sphere", Sphere(3, 0, 0)},
            {"noisy sphere", Sphere(3, 0.05, 7)},
        };
        for (const auto &[name, surface] : cases) {
            for (const bool quality : {false, true}) {
                const auto tets = tetra::Tetrahedralize(surface.Points, surface.Tris, {.Quality = quality});
                if (!tets) {
                    expect(false) << name << "tetrahedralization failed:" << tets.error();
                    continue;
                }
                const auto err = ValidateTetMesh(surface.Points, surface.Tris, tets->Mesh);
                expect(err.empty()) << name << "invalid tet mesh:" << err;
            }
        }
    };

    // Real-world complexity: a thin-walled RealImpact scan through the app's tet generation,
    // timing the solve. Skipped when the dataset is not present.
    "RealImpact bowl solves in reasonable time"_test = [] {
        const auto path = DatasetDir() / "9_BowlCeramic/preprocessed/transformed.obj";
        if (!std::filesystem::exists(path)) {
            std::println("skipping RealImpact benchmark: {} not found", path.string());
            return;
        }
        auto surface = LoadObj(path);
        if (!surface || surface->Positions.empty()) {
            std::println("skipping RealImpact benchmark: no mesh data in {}", path.string());
            return;
        }
        const std::vector<vec3> excite{surface->Positions.front()};
        const auto n_verts = surface->Positions.size(), n_tris = surface->TriangleIndices.size() / 3;
        const auto tets = GenerateTets(std::move(surface->Positions), std::move(surface->TriangleIndices), {});
        expect(tets.has_value());
        if (!tets) return;
        std::println("--- RealImpact bowl: {} welded verts, {} tris -> {} tet nodes, {} tets ---", n_verts, n_tris, tets->Mesh.Points.size(), tets->Mesh.Tets.size());

        constexpr AcousticMaterialProperties Ceramic{.Density = 2700, .YoungModulus = 7.2e10, .PoissonRatio = 0.19, .Alpha = 5, .Beta = 1e-8};
        const auto start = std::chrono::steady_clock::now();
        const auto result = modal::mesh2modes(tets->Mesh, Ceramic, excite, vec3{1}, {});
        const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        expect(!result.Modes.Freqs.empty());
        std::println("{:6.2f} s, {} modes, f1 {:8.1f} Hz", seconds, result.Modes.Freqs.size(), result.Modes.Freqs.empty() ? 0.0 : double(result.Modes.Freqs.front()));
    };

    // Every RealImpact object must tetrahedralize to a structurally valid mesh at every
    // resolution the app's simplification slider offers: simplifying a watertight, non-self-
    // intersecting surface should keep it tetrahedralizable. Skipped when the dataset is absent.
    "RealImpact meshes tetrahedralize to valid meshes across resolutions"_test = [] {
        const auto dataset = DatasetDir();
        if (!std::filesystem::exists(dataset)) {
            std::println("skipping RealImpact validation: {} not found", dataset.string());
            return;
        }
        std::vector<std::filesystem::path> objs;
        for (const auto &entry : std::filesystem::directory_iterator{dataset})
            if (entry.is_directory() && std::filesystem::exists(entry.path() / "preprocessed" / "transformed.obj"))
                objs.push_back(entry.path() / "preprocessed" / "transformed.obj");
        std::ranges::sort(objs);
        expect(!objs.empty());
        for (const auto &path : objs) {
            const auto name = path.parent_path().parent_path().filename().string();
            const auto surface = LoadObj(path);
            if (!surface || surface->Positions.empty()) {
                expect(false) << name << "failed to load";
                continue;
            }
            for (const float ratio : {1.0f, 0.5f, 0.25f}) {
                auto positions = surface->Positions;
                auto triangle_indices = surface->TriangleIndices;
                SimplifySurface(positions, triangle_indices, ratio);
                const std::vector<dvec3> in_points(positions.begin(), positions.end());
                const auto tets = GenerateTets(positions, triangle_indices, {});
                if (!tets) {
                    expect(false) << name << "@" << ratio << "tetrahedralization failed:" << tets.error();
                    continue;
                }
                const auto err = ValidateTetMesh(in_points, triangle_indices, tets->Mesh);
                expect(err.empty()) << name << "@" << ratio << "invalid tet mesh:" << err;
            }
        }
    };

    return RunSuites();
}
