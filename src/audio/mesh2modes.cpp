// Run Eigen's large dense products (Lanczos orthogonalization) on Accelerate's BLAS.
#define EIGEN_USE_BLAS

#include "mesh2modes.h"

#include "AcousticMaterialProperties.h"
#include "CholeskyShiftInvert.h"
#include "numeric/vec3.h"

#include "tetgen.h"
#include <Eigen/Eigenvalues>
#include <Spectra/SymGEigsShiftSolver.h>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/norm.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <limits>
#include <unordered_map>

using uint = uint32_t;

namespace {
double SecondsSince(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

// Runs `compute`, storing its wall-clock duration in `seconds`.
auto Timed(double &seconds, auto &&compute) {
    const auto start = std::chrono::steady_clock::now();
    auto result = compute();
    seconds = SecondsSince(start);
    return result;
}

int GetVertexIndex(const tetgenio &tets, uint element, uint vertex) { return tets.tetrahedronlist[element * 4 + vertex]; }

const glm::dvec3 &GetVertex(const tetgenio &tets, int element, int vertex) {
    return reinterpret_cast<const glm::dvec3 &>(tets.pointlist[3 * GetVertexIndex(tets, element, vertex)]);
}

double GetTetDeterminant(const dvec3 &a, const dvec3 &b, const dvec3 &c, const dvec3 &d) {
    return glm::dot(d - a, glm::cross(b - a, c - a));
}
double GetTetVolume(const dvec3 &a, const dvec3 &b, const dvec3 &c, const dvec3 &d) {
    return (1.f / 6.f) * fabs(GetTetDeterminant(a, b, c, d));
}
double GetElementDeterminant(const tetgenio &tets, int el) {
    return GetTetDeterminant(GetVertex(tets, el, 0), GetVertex(tets, el, 1), GetVertex(tets, el, 2), GetVertex(tets, el, 3));
}

// Lumped-vertex rigid-body mass properties in SI at the baked size: each tet's volume splits evenly onto its four
// vertices as point masses. `scale` maps tet coordinates to node-local, `length_to_si` maps node-local lengths to meters.
MassProperties ComputeMassProperties(const tetgenio &tets, double density, vec3 scale, double length_to_si) {
    const int nverts = tets.numberofpoints;
    const dvec3 inv_scale{1.0 / scale.x, 1.0 / scale.y, 1.0 / scale.z};
    std::vector<dvec3> pos(nverts);
    for (int i = 0; i < nverts; ++i) pos[i] = reinterpret_cast<const dvec3 &>(tets.pointlist[3 * i]) * inv_scale;

    std::vector<double> vol(nverts, 0.0);
    for (int el = 0; el < tets.numberoftetrahedra; ++el) {
        const int *t = &tets.tetrahedronlist[el * 4];
        const double quarter = GetTetVolume(pos[t[0]], pos[t[1]], pos[t[2]], pos[t[3]]) * 0.25;
        for (int c = 0; c < 4; ++c) vol[t[c]] += quarter;
    }

    double total = 0;
    dvec3 com{0};
    for (int i = 0; i < nverts; ++i) {
        total += vol[i];
        com += vol[i] * pos[i];
    }
    if (total <= 0) return {};
    com /= total;

    // Point-mass inertia about the center of mass, sum of vol * (|r|^2 I - r r^T), scaled to SI (inertia integral ~ length^5).
    const double s = length_to_si;
    Eigen::Matrix3d inertia = Eigen::Matrix3d::Zero();
    for (int i = 0; i < nverts; ++i) {
        const dvec3 r = pos[i] - com;
        const double rr = glm::dot(r, r);
        inertia(0, 0) += vol[i] * (rr - r.x * r.x);
        inertia(1, 1) += vol[i] * (rr - r.y * r.y);
        inertia(2, 2) += vol[i] * (rr - r.z * r.z);
        inertia(0, 1) -= vol[i] * r.x * r.y;
        inertia(0, 2) -= vol[i] * r.x * r.z;
        inertia(1, 2) -= vol[i] * r.y * r.z;
    }
    inertia(1, 0) = inertia(0, 1);
    inertia(2, 0) = inertia(0, 2);
    inertia(2, 1) = inertia(1, 2);
    inertia *= density * s * s * s * s * s;

    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(inertia);
    const auto &evals = es.eigenvalues();
    const auto &evecs = es.eigenvectors();
    glm::mat3 axes;
    for (int c = 0; c < 3; ++c)
        for (int r = 0; r < 3; ++r) axes[c][r] = float(evecs(r, c));
    if (glm::determinant(axes) < 0) axes[0] = -axes[0]; // ensure a proper rotation for the quaternion

    return {
        density * total * s * s * s,
        vec3{com},
        vec3{float(evals[0]), float(evals[1]), float(evals[2])},
        glm::normalize(glm::quat_cast(axes)),
    };
}

constexpr uint NumTetElementVertices{4};
constexpr uint NEV = NumTetElementVertices; // convenience

// Per-element volume and linear basis-function gradients.
struct ElementBasis {
    double Volume;
    dvec3 Phig[NEV]; // gradient of each corner's basis function
};

std::vector<ElementBasis> ComputeElementBases(const tetgenio &tets) {
    std::vector<ElementBasis> elements(tets.numberoftetrahedra);
    dvec3 columns[2];
    for (uint el = 0; el < elements.size(); ++el) {
        auto &element = elements[el];
        const double det = GetElementDeterminant(tets, el);
        element.Volume = fabs(det / 6);
        for (uint i = 0; i < NEV; ++i) {
            for (uint j = 0; j < 3; ++j) {
                uint ni = 0;
                for (uint ii = 0; ii < NEV; ++ii) {
                    if (ii == i) continue;

                    uint nj = 0;
                    for (uint jj = 0; jj < 3; ++jj) {
                        if (jj != j) {
                            columns[nj][ni] = GetVertex(tets, el, ii)[jj];
                            nj++;
                        }
                    }
                    ++ni;
                }
                const int sign = (i + j) % 2 == 0 ? -1 : 1;
                element.Phig[i][j] = sign * dot(dvec3{1}, cross(columns[0], columns[1])) / det;
            }
        }
    }
    return elements;
}

/***** Quadratic (10-node) tetrahedral elements *****/

// Polynomial in barycentric coordinates: a sum of c * l0^e0 * l1^e1 * l2^e2 * l3^e3 terms.
struct BaryTerm {
    double Coeff;
    std::array<int, 4> Exp;
};
using BaryPoly = std::vector<BaryTerm>;

BaryPoly Multiply(const BaryPoly &a, const BaryPoly &b) {
    BaryPoly product;
    product.reserve(a.size() * b.size());
    for (const auto &ta : a) {
        for (const auto &tb : b) {
            product.push_back({ta.Coeff * tb.Coeff, {ta.Exp[0] + tb.Exp[0], ta.Exp[1] + tb.Exp[1], ta.Exp[2] + tb.Exp[2], ta.Exp[3] + tb.Exp[3]}});
        }
    }
    return product;
}

// Integral over a straight-sided tet, divided by its volume: int l^e dV = 6V * prod(e!) / (sum(e) + 3)!
double UnitIntegral(const BaryPoly &p) {
    static constexpr double Factorial[]{1, 1, 2, 6, 24, 120, 720, 5040};
    double sum = 0;
    for (const auto &t : p) {
        const auto &e = t.Exp;
        sum += t.Coeff * 6 * Factorial[e[0]] * Factorial[e[1]] * Factorial[e[2]] * Factorial[e[3]] / Factorial[e[0] + e[1] + e[2] + e[3] + 3];
    }
    return sum;
}

constexpr uint NumQuadNodes{10};
// Local edge nodes 4-9 sit at the midpoints of these corner pairs.
constexpr uint EdgeCorners[6][2]{{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};

// Exact unit-volume integrals of the 10-node shape functions: corners N_i = l_i(2l_i - 1),
// edges N_ij = 4 l_i l_j. All integrals are polynomial, so the factorial formula is exact.
struct QuadBasis {
    double Mass[NumQuadNodes][NumQuadNodes]; // int N_a N_b dV / V
    double Grad[NumQuadNodes][4][NumQuadNodes][4]; // int (dN_a/dl_k)(dN_b/dl_l) dV / V
};

const QuadBasis &GetQuadBasis() {
    static const QuadBasis basis = [] {
        std::array<BaryPoly, NumQuadNodes> n;
        std::array<std::array<BaryPoly, 4>, NumQuadNodes> dn;
        for (int i = 0; i < 4; ++i) {
            n[i] = {{2, {2 * (i == 0), 2 * (i == 1), 2 * (i == 2), 2 * (i == 3)}}, {-1, {i == 0, i == 1, i == 2, i == 3}}};
            dn[i][i] = {{4, {i == 0, i == 1, i == 2, i == 3}}, {-1, {0, 0, 0, 0}}};
        }
        for (uint e = 0; e < 6; ++e) {
            const int i = EdgeCorners[e][0], j = EdgeCorners[e][1];
            n[4 + e] = {{4, {i == 0 || j == 0, i == 1 || j == 1, i == 2 || j == 2, i == 3 || j == 3}}};
            dn[4 + e][i] = {{4, {j == 0, j == 1, j == 2, j == 3}}};
            dn[4 + e][j] = {{4, {i == 0, i == 1, i == 2, i == 3}}};
        }
        QuadBasis b;
        for (uint a = 0; a < NumQuadNodes; ++a) {
            for (uint c = 0; c < NumQuadNodes; ++c) {
                b.Mass[a][c] = UnitIntegral(Multiply(n[a], n[c]));
                for (int k = 0; k < 4; ++k) {
                    for (int l = 0; l < 4; ++l) {
                        b.Grad[a][k][c][l] = dn[a][k].empty() || dn[c][l].empty() ? 0 : UnitIntegral(Multiply(dn[a][k], dn[c][l]));
                    }
                }
            }
        }
        return b;
    }();
    return basis;
}

// Global node ids of each element's 10 nodes: the 4 corners, then unique midside ids per edge,
// numbered after all corner nodes. Midside coordinates stay implicit (straight-sided elements).
struct QuadMesh {
    std::vector<std::array<uint, NumQuadNodes>> ElementNodes;
    uint NodeCount;
};

QuadMesh BuildQuadMesh(const tetgenio &tets) {
    QuadMesh quad;
    quad.ElementNodes.resize(tets.numberoftetrahedra);
    quad.NodeCount = tets.numberofpoints;
    std::unordered_map<uint64_t, uint> edge_nodes;
    edge_nodes.reserve(tets.numberoftetrahedra * 2);
    for (uint el = 0; el < uint(tets.numberoftetrahedra); ++el) {
        auto &nodes = quad.ElementNodes[el];
        for (uint c = 0; c < 4; ++c) nodes[c] = GetVertexIndex(tets, el, c);
        for (uint e = 0; e < 6; ++e) {
            const uint a = nodes[EdgeCorners[e][0]], b = nodes[EdgeCorners[e][1]];
            const uint64_t key = (uint64_t(std::min(a, b)) << 32) | std::max(a, b);
            const auto [it, inserted] = edge_nodes.try_emplace(key, quad.NodeCount);
            if (inserted) ++quad.NodeCount;
            nodes[4 + e] = it->second;
        }
    }
    return quad;
}

struct MassStiffness {
    Eigen::SparseMatrix<double> Mass, Stiffness;
};

// Isotropic linear-elastic mass and stiffness over 10-node elements. Basis gradients in physical
// coordinates are dN_a/dx = sum_k (dN_a/dl_k) grad(l_k), with grad(l_k) the linear-tet gradients (Phig).
// Only the lower triangle is filled (the eigensolver reads matrices as self-adjoint).
MassStiffness AssembleQuadratic(const tetgenio &tets, const QuadMesh &quad, const AcousticMaterialProperties &material) {
    const auto &basis = GetQuadBasis();
    const auto coeffs = ComputeElementBases(tets);
    const double lambda = material.Lambda();
    const double mu = material.Mu();

    std::vector<Eigen::Triplet<double>> mass_triplets, stiffness_triplets;
    mass_triplets.reserve(quad.ElementNodes.size() * NumQuadNodes * (NumQuadNodes + 1) / 2 * 3);
    stiffness_triplets.reserve(quad.ElementNodes.size() * NumQuadNodes * (NumQuadNodes + 1) / 2 * 9);
    for (uint el = 0; el < quad.ElementNodes.size(); ++el) {
        const auto &ed = coeffs[el];
        const auto &nodes = quad.ElementNodes[el];
        // Per-element gradient outer products: Outer[k][l][p][q] = Phig[k][p] * Phig[l][q].
        double outer[4][4][3][3];
        for (int k = 0; k < 4; ++k) {
            for (int l = 0; l < 4; ++l) {
                for (uint p = 0; p < 3; ++p) {
                    for (uint q = 0; q < 3; ++q) outer[k][l][p][q] = ed.Phig[k][p] * ed.Phig[l][q];
                }
            }
        }
        for (uint a = 0; a < NumQuadNodes; ++a) {
            for (uint c = 0; c < NumQuadNodes; ++c) {
                const uint row = 3 * nodes[a], col = 3 * nodes[c];
                if (row < col) continue;

                const double m = material.Density * ed.Volume * basis.Mass[a][c];
                for (uint k = 0; k < 3; ++k) mass_triplets.emplace_back(row + k, col + k, m);

                // G[p][q] = int (dN_a/dx_p)(dN_c/dx_q) dV / V
                double g[3][3]{};
                for (int k = 0; k < 4; ++k) {
                    for (int l = 0; l < 4; ++l) {
                        const double w = basis.Grad[a][k][c][l];
                        if (w == 0) continue;
                        for (uint p = 0; p < 3; ++p) {
                            for (uint q = 0; q < 3; ++q) g[p][q] += w * outer[k][l][p][q];
                        }
                    }
                }
                const double trace = g[0][0] + g[1][1] + g[2][2];
                for (uint p = 0; p < 3; ++p) {
                    for (uint q = 0; q < (row == col ? p + 1 : 3u); ++q) {
                        stiffness_triplets.emplace_back(row + p, col + q, ed.Volume * (lambda * g[p][q] + mu * g[q][p] + (p == q ? mu * trace : 0)));
                    }
                }
            }
        }
    }
    const uint n = 3 * quad.NodeCount;
    MassStiffness ms{Eigen::SparseMatrix<double>{n, n}, Eigen::SparseMatrix<double>{n, n}};
    ms.Mass.setFromTriplets(mass_triplets.begin(), mass_triplets.end());
    ms.Stiffness.setFromTriplets(stiffness_triplets.begin(), stiffness_triplets.end());
    return ms;
}

struct ComputeModesOpts {
    modal::SolverConfig Config{};
    std::vector<uint> ExPos{}; // Excitation positions
    std::vector<vec3> Positions{}; // Node-local positions of each excitation position, parallel to ExPos
    AcousticMaterialProperties Material{};
};

ModalModes ComputeModes(
    const Eigen::SparseMatrix<double> &M,
    const Eigen::SparseMatrix<double> &K,
    uint num_vertices, uint vertex_dim,
    ComputeModesOpts opts,
    modal::SolveProfile &profile
) {
    /** Compute mass/stiffness eigenvalues and eigenvectors **/
    using OpType = CholeskyShiftInvert;
    using BOpType = Spectra::SparseSymMatProd<double>;
    using Spectra::GEigsMode::ShiftInvert;

    const auto &config = opts.Config;
    const uint n = num_vertices * vertex_dim;
    const uint fem_n_modes = std::min(config.NumFemModes, n - 1);
    const uint basis_size = std::min(std::max(2 * fem_n_modes + 1, 20u), n); // Lanczos basis vector count (ncv)
    // Negative shift: K - sigma*M is positive definite, and the smallest
    // eigenvalues sit nearest the shift, so they still converge first.
    const double sigma = -pow(2 * M_PI * config.MinModeFreq, 2);
    OpType op{K, M, profile.Factorize, profile.OpSolve};
    BOpType Bop{M};
    const auto eig_start = std::chrono::steady_clock::now();
    // The solver factorizes K - sigma*M in its constructor (via set_shift).
    Spectra::SymGEigsShiftSolver<OpType, BOpType, ShiftInvert> eigs{op, Bop, fem_n_modes, basis_size, sigma};
    eigs.init();
    eigs.compute(Spectra::SortRule::LargestMagn, config.MaxRestarts, config.Tolerance, Spectra::SortRule::SmallestAlge);
    profile.Iterate = SecondsSince(eig_start) - profile.Factorize;
    profile.OpApplications = eigs.num_operations();
    profile.Restarts = eigs.num_iterations();
    if (eigs.info() != Spectra::CompInfo::Successful) return {};

    const auto eigenvalues = eigs.eigenvalues();

    /** Compute modes frequencies/gains/T60s **/
    std::vector<float> mode_freqs(fem_n_modes), mode_t60s(fem_n_modes);
    std::vector<double> omega_undamped(fem_n_modes);
    const double lambda_eps = std::fabs(sigma) * 1e-10; // scale-aware near-zero cutoff
    for (uint mode = 0; mode < fem_n_modes; ++mode) {
        const double lambda_i = eigenvalues[mode];
        omega_undamped[mode] = lambda_i > lambda_eps ? std::sqrt(lambda_i) : 0;
    }

    const auto c_from_omega = [&material = opts.Material](double omega) { return material.Alpha + material.Beta * (omega * omega); };
    const auto damped_hz = [&](double omega, double c) {
        const double omega_d_sq = omega * omega - 0.25 * c * c;
        return omega_d_sq > 0 ? std::sqrt(omega_d_sq) / (2 * M_PI) : 0;
    };

    uint lowest_mode_i = fem_n_modes;
    float lowest_mode_freq_orig{0};
    for (uint mode = 0; mode < fem_n_modes; ++mode) {
        const double omega_i = omega_undamped[mode];
        if (omega_i <= 0) {
            mode_freqs[mode] = mode_t60s[mode] = 0.f;
            continue;
        }
        mode_freqs[mode] = damped_hz(omega_i, c_from_omega(omega_i));
        // Rigid-body modes carry numerically tiny but nonzero eigenvalues, so a mode is valid
        // only at or above the audible floor.
        if (lowest_mode_i == fem_n_modes && mode_freqs[mode] >= config.MinModeFreq) {
            lowest_mode_i = mode;
            lowest_mode_freq_orig = mode_freqs[mode];
        }
    }
    if (lowest_mode_i == fem_n_modes) return {};

    // Scale all modes so the lowest valid mode is at the configured fundamental frequency,
    // and calculate T60s from the scaled frequencies.
    static const double ln_1000 = std::log(1000);
    const float freq_scale = config.FundamentalFreq ? *config.FundamentalFreq / lowest_mode_freq_orig : 1.f;
    for (uint mode = lowest_mode_i; mode < fem_n_modes; ++mode) {
        const double omega_s = omega_undamped[mode] * freq_scale; // scaled rad/s
        const double c = c_from_omega(omega_s);
        mode_freqs[mode] = damped_hz(omega_s, c);
        mode_t60s[mode] = c > 0 ? (2 * ln_1000) / c : 0;
    }
    // Keep modes that are only above the max frequency because of scaling.
    // This allows changing the fundamental without losing the higher modes.
    const float max_mode_freq = config.MaxModeFreq * std::max(1.f, freq_scale);
    uint highest_mode_i = fem_n_modes;
    while (highest_mode_i > lowest_mode_i && mode_freqs[highest_mode_i - 1] > max_mode_freq) --highest_mode_i;

    // Adjust modes to include only the requested range.
    const uint n_modes = std::min({config.NumModes, fem_n_modes, highest_mode_i - lowest_mode_i});
    mode_freqs.erase(mode_freqs.begin(), mode_freqs.begin() + lowest_mode_i);
    mode_freqs.resize(n_modes);
    mode_t60s.erase(mode_t60s.begin(), mode_t60s.begin() + lowest_mode_i);
    mode_t60s.resize(n_modes);

    // One mode shape 3-vector per excitable vertex, so `shapes` stays the same length as `ExPos`.
    // Spectra returns M-orthonormal eigenvectors, so these are already mass-normalized (kg^-1/2).
    const auto eigenvectors = Timed(profile.Extract, [&] { return eigs.eigenvectors(lowest_mode_i + n_modes); });
    std::vector<std::vector<vec3>> shapes(opts.ExPos.size()); // Mode shapes by [excitation position][mode]
    for (size_t ex_pos = 0; ex_pos < shapes.size(); ++ex_pos) {
        const uint ev_i = vertex_dim * opts.ExPos[ex_pos];
        shapes[ex_pos] = std::vector<vec3>(n_modes);
        for (uint mode = 0; mode < n_modes; ++mode) {
            for (uint vi = 0; vi < vertex_dim; ++vi) shapes[ex_pos][mode][vi] = eigenvectors(ev_i + vi, mode + lowest_mode_i);
        }
    }

    return {std::move(mode_freqs), std::move(mode_t60s), std::move(shapes), {}, std::move(opts.Positions), lowest_mode_freq_orig};
}
} // namespace

modal::ModalResult modal::mesh2modes(const tetgenio &tets, const AcousticMaterialProperties &material, const std::vector<vec3> &excite_positions, vec3 baked_scale, SolverConfig config) {
    SolveProfile profile;
    const double length_to_si = (double(baked_scale.x) + baked_scale.y + baked_scale.z) / 3.0;
    auto mass_props = Timed(profile.MassProps, [&] { return ComputeMassProperties(tets, material.Density, baked_scale, length_to_si); });

    const auto quad = Timed(profile.QuadMesh, [&] { return BuildQuadMesh(tets); });
    const auto [M, K] = Timed(profile.Assemble, [&] { return AssembleQuadratic(tets, quad, material); });
    profile.Dofs = 3 * quad.NodeCount;
    profile.StiffnessNonZeros = K.nonZeros();

    // Sample each excitation position at its nearest tet point, recovering node-local coordinates.
    auto [excite_points, positions] = Timed(profile.SampleExcite, [&] {
        const dvec3 inv_scale{1.0 / baked_scale.x, 1.0 / baked_scale.y, 1.0 / baked_scale.z};
        std::vector<uint> points(excite_positions.size());
        std::vector<vec3> local(excite_positions.size());
        for (size_t i = 0; i < excite_positions.size(); ++i) {
            const dvec3 p{excite_positions[i]};
            double best = std::numeric_limits<double>::max();
            for (int v = 0; v < tets.numberofpoints; ++v) {
                const auto &q = reinterpret_cast<const dvec3 &>(tets.pointlist[3 * v]);
                if (const double d = glm::distance2(p, q); d < best) {
                    best = d;
                    points[i] = v;
                }
            }
            local[i] = vec3{reinterpret_cast<const dvec3 &>(tets.pointlist[3 * points[i]]) * inv_scale};
        }
        return std::pair{std::move(points), std::move(local)};
    });
    auto modes = ComputeModes(M, K, quad.NodeCount, 3, {
                                                           .Config = std::move(config),
                                                           .ExPos = std::move(excite_points),
                                                           .Positions = std::move(positions),
                                                           .Material = material,
                                                       },
                              profile);
    return {std::move(modes), std::move(mass_props), profile};
}
