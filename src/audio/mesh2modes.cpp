// Run Eigen's large dense products (Lanczos orthogonalization) on Accelerate's BLAS.
#define EIGEN_USE_BLAS

#include "mesh2modes.h"

#include "AcousticMaterialProperties.h"
#include "CholeskyShiftInvert.h"
#include "Job.h"
#include "numeric/vec3.h"

#include "mesh/TetMesh.h"
#include <Eigen/Eigenvalues>
#include <Spectra/SymGEigsShiftSolver.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <limits>
#include <numbers>
#include <optional>
#include <random>
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

// Degenerate elements contribute nothing physically, but their inverse-determinant basis gradients poison the stiffness matrix. Copy the mesh without them.
TetMesh FilterDegenerate(const TetMesh &tets) {
    TetMesh clean;
    clean.Points = tets.Points;
    clean.Tets.reserve(tets.Tets.size());
    for (const auto &t : tets.Tets) {
        const dvec3 &a = tets.Points[t[0]];
        const dvec3 r0 = tets.Points[t[1]] - a, r1 = tets.Points[t[2]] - a, r2 = tets.Points[t[3]] - a;
        const double det = std::abs(numeric::Dot(r0, numeric::Cross(r1, r2)));
        double lmax_sq = 0;
        for (uint i = 0; i < 4; ++i) {
            for (uint j = i + 1; j < 4; ++j) {
                const dvec3 d = tets.Points[t[i]] - tets.Points[t[j]];
                lmax_sq = std::max(lmax_sq, numeric::Dot(d, d));
            }
        }
        if (det > 1e-12 * lmax_sq * std::sqrt(lmax_sq)) clean.Tets.push_back(t);
    }
    return clean;
}

const dvec3 &GetVertex(const TetMesh &tets, uint element, uint vertex) { return tets.Points[tets.Tets[element][vertex]]; }

double GetTetDeterminant(const dvec3 &a, const dvec3 &b, const dvec3 &c, const dvec3 &d) {
    return numeric::Dot(d - a, numeric::Cross(b - a, c - a));
}
double GetTetVolume(const dvec3 &a, const dvec3 &b, const dvec3 &c, const dvec3 &d) {
    return (1.f / 6.f) * fabs(GetTetDeterminant(a, b, c, d));
}

// Computes SI rigid-body mass properties by distributing each tetrahedron's volume equally across its vertices.
// scale maps tetrahedral coordinates to node-local space, and length_to_si converts node-local lengths to metres.
MassProperties ComputeMassProperties(const TetMesh &tets, double density, vec3 scale, double length_to_si) {
    const size_t nverts = tets.Points.size();
    const dvec3 inv_scale{1.0 / scale.x, 1.0 / scale.y, 1.0 / scale.z};
    std::vector<dvec3> pos(nverts);
    for (size_t i = 0; i < nverts; ++i) pos[i] = tets.Points[i] * inv_scale;

    std::vector<double> vol(nverts, 0.0);
    for (const auto &t : tets.Tets) {
        const double quarter = GetTetVolume(pos[t[0]], pos[t[1]], pos[t[2]], pos[t[3]]) * 0.25;
        for (int c = 0; c < 4; ++c) vol[t[c]] += quarter;
    }

    double total = 0;
    dvec3 com{0};
    for (size_t i = 0; i < nverts; ++i) {
        total += vol[i];
        com += vol[i] * pos[i];
    }
    if (total <= 0) return {};
    com /= total;

    // Point-mass inertia about the center of mass, sum of vol * (|r|^2 I - r r^T), scaled to SI (inertia integral ~ length^5).
    const double s = length_to_si;
    Eigen::Matrix3d inertia = Eigen::Matrix3d::Zero();
    for (size_t i = 0; i < nverts; ++i) {
        const dvec3 r = pos[i] - com;
        const double rr = numeric::Dot(r, r);
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
    mat3 axes;
    for (int c = 0; c < 3; ++c)
        for (int r = 0; r < 3; ++r) axes[c][r] = float(evecs(r, c));
    if (numeric::Determinant(axes) < 0) axes[0] = -axes[0]; // ensure a proper rotation for the quaternion

    return {
        density * total * s * s * s,
        vec3{com},
        vec3{float(evals[0]), float(evals[1]), float(evals[2])},
        numeric::Normalize(numeric::ToQuat(axes)),
    };
}

constexpr uint NumTetElementVertices{4};
constexpr uint NEV = NumTetElementVertices; // convenience

// Per-element volume and linear basis-function gradients.
struct ElementBasis {
    double Volume;
    dvec3 Phig[NEV]; // gradient of each corner's basis function
};

std::vector<ElementBasis> ComputeElementBases(const TetMesh &tets) {
    std::vector<ElementBasis> elements(tets.Tets.size());
    dvec3 columns[2];
    for (uint el = 0; el < elements.size(); ++el) {
        auto &element = elements[el];
        const double det = GetTetDeterminant(GetVertex(tets, el, 0), GetVertex(tets, el, 1), GetVertex(tets, el, 2), GetVertex(tets, el, 3));
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
                element.Phig[i][j] = sign * numeric::Dot(dvec3{1}, numeric::Cross(columns[0], columns[1])) / det;
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

// Integrates quadratic corner functions N_i = l_i(2l_i - 1) and edge functions N_ij = 4*l_i*l_j over unit volume.
// The factorial formula is exact for these polynomials.
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

// Numbers four corner nodes followed by unique midside edge nodes after all corners.
// Straight-sided elements keep midside coordinates implicit.
struct QuadMesh {
    std::vector<std::array<uint, NumQuadNodes>> ElementNodes;
    uint NodeCount;
};

QuadMesh BuildQuadMesh(const TetMesh &tets) {
    QuadMesh quad;
    quad.ElementNodes.resize(tets.Tets.size());
    quad.NodeCount = uint(tets.Points.size());
    std::unordered_map<uint64_t, uint> edge_nodes;
    edge_nodes.reserve(tets.Tets.size() * 2);
    for (uint el = 0; el < uint(tets.Tets.size()); ++el) {
        auto &nodes = quad.ElementNodes[el];
        for (uint c = 0; c < 4; ++c) nodes[c] = tets.Tets[el][c];
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

// Assembles isotropic linear-elastic mass and stiffness for ten-node elements.
// Physical gradients satisfy dN_a/dx = sum_k (dN_a/dl_k)*grad(l_k), where Phig contains linear-tetrahedron gradients.
// Only the lower triangle is filled (the eigensolver reads matrices as self-adjoint).
MassStiffness AssembleQuadratic(const TetMesh &tets, const QuadMesh &quad, const AcousticMaterialProperties &material) {
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

// Applies block subspace iteration to a shifted pencil from an optional prior eigenvector basis.
// Each iteration solves (K - sigma*M) Xbar = M X and applies Rayleigh-Ritz projection over span(Xbar).
// Returns the leading nev M-orthonormal eigenpairs in ascending order.
struct SubspaceResult {
    Eigen::VectorXd Eigenvalues; // ascending, size nev, empty when convergence failed
    Eigen::MatrixXd Eigenvectors; // n x nev, M-orthonormal
    uint Iterations{}, OpApplications{};
};

SubspaceResult SubspaceIterate(
    const CholeskyShiftInvert &op, const Eigen::SparseMatrix<double> &M,
    uint nev, uint p, double sigma, double tol, uint max_iters,
    const Eigen::MatrixXf &x0, // warm-start basis, its columns seed the leading panel columns
    JobMonitor *monitor = nullptr
) {
    const uint n = M.rows();
    const auto Msa = M.selfadjointView<Eigen::Lower>();

    Eigen::MatrixXd MX(n, p);
    {
        Eigen::MatrixXd X(n, p);
        std::mt19937_64 rng{20260710};
        std::normal_distribution<double> gauss;
        const uint seeded = std::min(uint(x0.cols()), p);
        X.leftCols(seeded) = x0.leftCols(seeded).cast<double>();
        for (uint j = seeded; j < p; ++j)
            for (uint i = 0; i < n; ++i) X(i, j) = gauss(rng);
        MX.noalias() = Msa * X;
    }

    SubspaceResult result;
    // Store converged leading Ritz pairs in ascending order.
    // Deflate the active block against stored vectors.
    Eigen::MatrixXd XL(n, nev), MXL(n, nev);
    Eigen::VectorXd theta_locked(nev);
    uint c = 0; // locked count

    Eigen::VectorXd prev_lambda = Eigen::VectorXd::Constant(nev, std::numeric_limits<double>::max());
    for (uint iter = 0; iter < max_iters; ++iter) {
        // A cancelled solve stops between block iterations, leaving the result empty.
        if (monitor && monitor->Cancelled()) return result;
        const uint w = p - c;
        Eigen::MatrixXd Xbar(n, w);
        op.solve_panel(MX.data(), Xbar.data(), int(w)); // (K - sigma*M) Xbar = M X
        result.OpApplications += w;

        // Kr = Xbar^T (K - sigma*M) Xbar = Xbar^T M X, corrected below for deflation.
        Eigen::MatrixXd Kr = Xbar.transpose() * MX;
        Eigen::MatrixXd MXbar = Msa * Xbar;

        // Deflate using the correction -C^T*theta*C from (K - sigma*M)*x = theta*M*x.
        if (c > 0) {
            const Eigen::MatrixXd C = XL.leftCols(c).transpose() * MXbar;
            Xbar.noalias() -= XL.leftCols(c) * C;
            MXbar.noalias() -= MXL.leftCols(c) * C;
            Kr.noalias() -= C.transpose() * theta_locked.head(c).asDiagonal() * C;
        }
        Eigen::MatrixXd Mr = Xbar.transpose() * MXbar;

        // Rescale columns to unit M-norm to keep the small GEVP well-conditioned.
        Kr = (0.5 * (Kr + Kr.transpose())).eval();
        Mr = (0.5 * (Mr + Mr.transpose())).eval();
        const Eigen::VectorXd dscale = Mr.diagonal().cwiseSqrt().cwiseInverse();
        Kr = dscale.asDiagonal() * Kr * dscale.asDiagonal();
        Mr = dscale.asDiagonal() * Mr * dscale.asDiagonal();
        const Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> es(Kr, Mr);
        if (es.info() != Eigen::Success) return result;

        const Eigen::MatrixXd q = dscale.asDiagonal() * es.eigenvectors();

        uint newly_locked = 0;
        for (uint i = 0; i < w && c + i < nev; ++i) {
            const double lambda = es.eigenvalues()[i] + sigma;
            const double rel = std::abs(lambda - prev_lambda[c + i]) / std::max(std::abs(lambda), std::abs(sigma));
            prev_lambda[c + i] = lambda;
            if (newly_locked == i && rel < tol) ++newly_locked;
        }
        if (newly_locked > 0) {
            XL.middleCols(c, newly_locked).noalias() = Xbar * q.leftCols(newly_locked);
            MXL.middleCols(c, newly_locked).noalias() = MXbar * q.leftCols(newly_locked);
            theta_locked.segment(c, newly_locked) = es.eigenvalues().head(newly_locked);
            c += newly_locked;
        }
        if (monitor) monitor->Progress.store(0.3f + 0.65f * float(c) / float(nev), std::memory_order_relaxed);
        result.Iterations = iter + 1;
        if (c >= nev) {
            result.Eigenvalues = prev_lambda;
            result.Eigenvectors = std::move(XL);
            return result;
        }
        // Rotate the maintained M X onto the remaining active Ritz vectors.
        MX.noalias() = MXbar * q.rightCols(w - newly_locked);
    }
    return result;
}

struct ComputeModesOpts {
    modal::SolverConfig Config{};
    std::vector<uint> ExPos{}; // Excitation positions
    std::vector<vec3> Positions{}; // Node-local positions of each excitation position, parallel to ExPos
    AcousticMaterialProperties Material{};
    modal::SolveReuse Reuse{};
    JobMonitor *Monitor{};
};

// `summary_out` receives the solved eigenpairs at the excitation positions, and `basis_out`
// (when non-null) the full eigenvector basis.
ModalModes ComputeModes(
    const Eigen::SparseMatrix<double> &M,
    const Eigen::SparseMatrix<double> &K,
    uint num_vertices, uint vertex_dim,
    ComputeModesOpts opts,
    modal::SolveProfile &profile,
    ModalEigenSummary &summary_out, Eigen::MatrixXf *basis_out
) {
    /** Compute mass/stiffness eigenvalues and eigenvectors **/
    using OpType = CholeskyShiftInvert;
    using BOpType = Spectra::SparseSymMatProd<double>;
    using Spectra::GEigsMode::ShiftInvert;

    const auto &config = opts.Config;
    const uint n = num_vertices * vertex_dim;
    const uint fem_n_modes = std::min(config.NumFemModes, n - 1);
    const uint basis_size = std::min(std::max(fem_n_modes + 20, 20u), n); // Lanczos basis vector count (ncv)
    // Negative shift: K - sigma*M is positive definite, and the smallest eigenvalues sit nearest the shift, so they still converge first.
    const double sigma = -pow(2 * std::numbers::pi * config.MinModeFreq, 2);
    auto *monitor = opts.Monitor;
    if (monitor && monitor->Cancelled()) return {};
    OpType op{K, M, profile.Factorize, profile.OpSolve};
    BOpType Bop{M};
    if (monitor) monitor->Progress.store(0.3f, std::memory_order_relaxed);
    // A basis solved over a different mesh cannot seed this solve, so it falls back to a cold solve.
    const auto &reuse = opts.Reuse;
    // Subspace iteration re-converges a warm-start basis in a few block iterations.
    const bool use_subspace = reuse.SeedBasis != nullptr && reuse.SeedBasis->rows() == Eigen::Index(n) && reuse.SeedBasis->cols() >= Eigen::Index(fem_n_modes);
    std::optional<Spectra::SymGEigsShiftSolver<OpType, BOpType, ShiftInvert>> eigs;
    SubspaceResult subspace;
    Eigen::VectorXd eigenvalues;
    const auto eig_start = std::chrono::steady_clock::now();
    if (use_subspace) {
        op.set_shift(sigma);
        subspace = SubspaceIterate(op, M, fem_n_modes, std::min(fem_n_modes + 15, n), sigma, config.WarmTolerance, config.MaxRestarts, *reuse.SeedBasis, monitor);
        profile.OpApplications = subspace.OpApplications;
        profile.Restarts = subspace.Iterations;
        if (subspace.Eigenvalues.size() == 0) return {};
        eigenvalues = subspace.Eigenvalues;
    } else {
        if (monitor && monitor->Cancelled()) return {};
        // The solver factorizes K - sigma*M in its constructor (via set_shift).
        // Spectra's compute runs to completion, so a cold solve reports no progress and cancels only at stage boundaries.
        eigs.emplace(op, Bop, fem_n_modes, basis_size, sigma);
        eigs->init();
        eigs->compute(Spectra::SortRule::LargestMagn, config.MaxRestarts, config.Tolerance, Spectra::SortRule::SmallestAlge);
        profile.OpApplications = eigs->num_operations();
        profile.Restarts = eigs->num_iterations();
        if (eigs->info() != Spectra::CompInfo::Successful) return {};
        eigenvalues = eigs->eigenvalues();
    }
    profile.Iterate = SecondsSince(eig_start) - profile.Factorize;
    // The eigenvectors are M-orthonormal, so shapes are already mass-normalized (kg^-1/2).
    const Eigen::MatrixXd eigenvectors = Timed(profile.Extract, [&]() -> Eigen::MatrixXd {
        return use_subspace ? std::move(subspace.Eigenvectors) : eigs->eigenvectors();
    });
    std::vector<std::vector<vec3>> shapes(opts.ExPos.size(), std::vector<vec3>(fem_n_modes));
    for (size_t ex_pos = 0; ex_pos < shapes.size(); ++ex_pos) {
        const uint ev_i = vertex_dim * opts.ExPos[ex_pos];
        for (uint mode = 0; mode < fem_n_modes; ++mode) {
            for (uint vi = 0; vi < vertex_dim; ++vi) shapes[ex_pos][mode][vi] = eigenvectors(ev_i + vi, mode);
        }
    }
    summary_out.Eigenvalues = {eigenvalues.begin(), eigenvalues.end()};
    summary_out.Shapes = shapes;
    summary_out.SolvedMaterial = opts.Material;
    if (basis_out) *basis_out = eigenvectors.cast<float>();

    return modal::PostprocessModes(summary_out.Eigenvalues, shapes, 1.f, opts.Material, config, std::move(opts.Positions));
}
} // namespace

ModalModes modal::PostprocessModes(std::span<const double> eigenvalues, const std::vector<std::vector<vec3>> &shapes, float shape_scale, const AcousticMaterialProperties &material, const SolverConfig &config, std::vector<vec3> positions) {
    const uint fem_n_modes = eigenvalues.size();
    std::vector<float> mode_freqs(fem_n_modes), mode_t60s(fem_n_modes);
    std::vector<double> omega_undamped(fem_n_modes);
    // Scale-aware near-zero cutoff, relative to the eigensolver shift.
    const double lambda_eps = pow(2 * std::numbers::pi * config.MinModeFreq, 2) * 1e-10;
    for (uint mode = 0; mode < fem_n_modes; ++mode) {
        const double lambda_i = eigenvalues[mode];
        omega_undamped[mode] = lambda_i > lambda_eps ? std::sqrt(lambda_i) : 0;
    }

    const auto c_from_omega = [&material](double omega) { return material.Alpha + material.Beta * (omega * omega); };
    const auto damped_hz = [&](double omega, double c) {
        const double omega_d_sq = omega * omega - 0.25 * c * c;
        return omega_d_sq > 0 ? std::sqrt(omega_d_sq) / (2 * std::numbers::pi) : 0;
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
        // Rigid-body modes can have numerically tiny positive eigenvalues, so a mode is valid only at or above the audible floor.
        if (lowest_mode_i == fem_n_modes && mode_freqs[mode] >= config.MinModeFreq) {
            lowest_mode_i = mode;
            lowest_mode_freq_orig = mode_freqs[mode];
        }
    }
    if (lowest_mode_i == fem_n_modes) return {};

    // Scale all modes so the lowest valid mode is at the configured fundamental frequency, and calculate T60s from the scaled frequencies.
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

    // One mode shape 3-vector per excitable vertex, so `out_shapes` stays the same length as `shapes`.
    std::vector<std::vector<vec3>> out_shapes(shapes.size(), std::vector<vec3>(n_modes));
    for (size_t ex_pos = 0; ex_pos < shapes.size(); ++ex_pos) {
        for (uint mode = 0; mode < n_modes; ++mode) out_shapes[ex_pos][mode] = shapes[ex_pos][mode + lowest_mode_i] * shape_scale;
    }

    return {
        .Freqs = std::move(mode_freqs),
        .T60s = std::move(mode_t60s),
        .Shapes = std::move(out_shapes),
        .Vertices = {},
        .Positions = std::move(positions),
        .Indices = {},
        .OriginalFundamentalFreq = lowest_mode_freq_orig,
    };
}

std::optional<ModalModes> modal::RescaleModes(const ModalEigenSummary &summary, const ModalModes &current, const AcousticMaterialProperties &material, SolverConfig config) {
    if (summary.Eigenvalues.empty() || material.PoissonRatio != summary.SolvedMaterial.PoissonRatio) return {};

    const double rho_ratio = material.Density / summary.SolvedMaterial.Density;
    const double eigenvalue_scale = (material.YoungModulus / summary.SolvedMaterial.YoungModulus) / rho_ratio;
    auto eigenvalues = summary.Eigenvalues;
    for (auto &v : eigenvalues) v *= eigenvalue_scale;

    auto modes = PostprocessModes(eigenvalues, summary.Shapes, float(1 / std::sqrt(rho_ratio)), material, config, current.Positions);
    modes.Vertices = current.Vertices;
    modes.Indices = current.Indices;
    modes.BakedScale = current.BakedScale;
    return modes;
}

modal::ModalResult modal::mesh2modes(const TetMesh &input_tets, const AcousticMaterialProperties &material, const std::vector<vec3> &excite_positions, vec3 baked_scale, SolverConfig config, SolveReuse reuse, JobMonitor *monitor) {
    const TetMesh tets = FilterDegenerate(input_tets);
    SolveProfile profile;
    const double length_to_si = (double(baked_scale.x) + baked_scale.y + baked_scale.z) / 3.0;
    auto mass_props = Timed(profile.MassProps, [&] { return ComputeMassProperties(tets, material.Density, baked_scale, length_to_si); });

    if (monitor) monitor->Progress.store(0.1f, std::memory_order_relaxed);
    const auto quad = Timed(profile.QuadMesh, [&] { return BuildQuadMesh(tets); });
    const auto [M, K] = Timed(profile.Assemble, [&] { return AssembleQuadratic(tets, quad, material); });
    profile.Dofs = 3 * quad.NodeCount;
    profile.StiffnessNonZeros = K.nonZeros();
    if (monitor && monitor->Cancelled()) return {};

    auto [excite_points, positions, sample_point_of] = Timed(profile.SampleExcite, [&] {
        const dvec3 inv_scale{1.0 / baked_scale.x, 1.0 / baked_scale.y, 1.0 / baked_scale.z};
        std::vector<uint> points;
        std::vector<vec3> local;
        std::vector<uint32_t> remap(excite_positions.size());
        std::unordered_map<uint, uint32_t> sample_point_at; // Tet point to the sample point that first reached it.
        for (size_t i = 0; i < excite_positions.size(); ++i) {
            const dvec3 p{excite_positions[i]};
            double best = std::numeric_limits<double>::max();
            uint nearest = 0;
            for (uint v = 0; v < uint(tets.Points.size()); ++v) {
                const auto &q = tets.Points[v];
                if (const double d = numeric::Distance2(p, q); d < best) {
                    best = d;
                    nearest = v;
                }
            }
            const auto [entry, first] = sample_point_at.emplace(nearest, uint32_t(points.size()));
            if (first) {
                points.push_back(nearest);
                local.emplace_back(tets.Points[nearest] * inv_scale);
            }
            remap[i] = entry->second;
        }
        return std::tuple{std::move(points), std::move(local), std::move(remap)};
    });
    ModalEigenSummary summary;
    Eigen::MatrixXf basis;
    auto modes = ComputeModes(M, K, quad.NodeCount, 3, {
                                                           .Config = std::move(config),
                                                           .ExPos = std::move(excite_points),
                                                           .Positions = std::move(positions),
                                                           .Material = material,
                                                           .Reuse = reuse,
                                                           .Monitor = monitor,
                                                       },
                              profile, summary, reuse.KeepBasis ? &basis : nullptr);
    return {std::move(modes), std::move(mass_props), profile, std::move(summary), std::move(basis), std::move(sample_point_of)};
}
