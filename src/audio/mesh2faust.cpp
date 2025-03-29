#include "mesh2faust.h"

#include "AcousticMaterialProperties.h"
#include "numeric/mat3.h"
#include "numeric/vec3.h"

#include "tetgen.h"
#include <Eigen/SparseCore>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/MatOp/SymShiftInvert.h>
#include <Spectra/SymGEigsShiftSolver.h>

#include <ranges>

using std::ranges::find_if, std::views::drop, std::views::reverse;

using uint = uint32_t;

namespace {

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
double GetElementVolume(const tetgenio &tets, int el) {
    return GetTetVolume(GetVertex(tets, el, 0), GetVertex(tets, el, 1), GetVertex(tets, el, 2), GetVertex(tets, el, 3));
}

constexpr uint NumTetElementVertices{4};
constexpr uint NEV = NumTetElementVertices; // convenience

using ElementMatrix = double[NEV][NEV];

struct ElementData {
    double volume;
    dvec3 Phig[NEV]; // gradient of a basis function

    dmat3 A(uint i, uint j) const { return volume * glm::outerProduct(Phig[j], Phig[i]); }
    double B(uint i, uint j, const ElementMatrix &dots) const { return volume * dots[i][j]; }
    dvec3 C(uint i, uint j, uint k, const ElementMatrix &dots) const { return volume * dots[j][k] * Phig[i]; }
    double D(uint i, uint j, uint k, uint l, const ElementMatrix &dots) const { return volume * dots[i][j] * dots[k][l]; }
};

// Computes the St.Venant-Kirchhoff coefficients for each tetrahedral element.
std::vector<ElementData> StVKABCD(const tetgenio &tets) {
    std::vector<ElementData> elements_data(tets.numberoftetrahedra);
    dvec3 columns[2];
    for (uint el = 0; el < elements_data.size(); ++el) {
        auto &element_data = elements_data[el];
        // Create the element data structure for a tet
        const double det = GetElementDeterminant(tets, el);
        element_data.volume = fabs(det / 6);
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
                    const int sign = (i + j) % 2 == 0 ? -1 : 1;
                    element_data.Phig[i][j] = sign * dot(dvec3{1}, cross(columns[0], columns[1])) / det;
                    ++ni;
                }
            }
        }
    }
    return elements_data;
}

// Compute the tangent stiffness matrix of a StVK elastic deformable object.
// As a special case, the routine can compute the stiffness matrix in the rest configuration.
// `vertexDisplacements` is an array of vertex deformations, of length 3*n, where n is the total number of mesh vertices.
// (Since we assume displacements are small for linear modal analysis, we neglect displacement terms.)
Eigen::SparseMatrix<double> ComputeStiffnessMatrix(const tetgenio &tets, const AcousticMaterialProperties &material /*, const double *vertex_displacements*/) {
    const uint ne = tets.numberoftetrahedra;
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(ne * NEV * NEV * 9);

    const double lambda = material.Lambda();
    const double mu = material.Mu();
    const auto coeffs = StVKABCD(tets);
    ElementMatrix dots; // Cache dot products between basis gradients for current element
    for (uint el = 0; el < ne; ++el) {
        const auto &ed = coeffs[el];
        for (uint i = 0; i < NEV; ++i) {
            for (uint j = 0; j < NEV; ++j) {
                dots[i][j] = dot(ed.Phig[i], ed.Phig[j]);
            }
        }
        for (uint c = 0; c < NEV; ++c) {
            for (uint a = 0; a < NEV; ++a) {
                /*
                const auto &qa = reinterpret_cast<const dvec3 &>(vertex_displacements[3 * GetVertexIndex(tets, el, a)]);
                // Quadratic terms
                dmat3 quad_mat{0};
                {
                    for (uint e = 0; e < NEV; e++) {
                        const dvec3 C0 = lambda * ed.C(c, a, e, dots) + mu * (ed.C(e, a, c, dots) + ed.C(a, e, c, dots));
                        const dvec3 C1 = lambda * ed.C(e, a, c, dots) + mu * (ed.C(c, e, a, dots) + ed.C(a, e, c, dots));
                        const dvec3 C2 = lambda * ed.C(a, e, c, dots) + mu * (ed.C(c, a, e, dots) + ed.C(e, a, c, dots));
                        quad_mat += glm::outerProduct(C0, qa) + glm::outerProduct(qa, C1) + dmat3{dot(qa, C2)};
                    }
                }
                // Cubic terms
                dmat3 cubic_mat{0};
                {
                    for (uint b = 0; b < NEV; b++) {
                        const auto &qb = reinterpret_cast<const dvec3 &>(vertex_displacements[3 * GetVertexIndex(tets, el, b)]);
                        for (uint e = 0; e < NEV; e++) {
                            const double D0 = lambda * ed.D(a, c, b, e, dots) + mu * (ed.D(a, e, b, c, dots) + ed.D(a, b, c, e, dots));
                            const double D1 = 0.5 * lambda * ed.D(a, b, c, e, dots) + mu * ed.D(a, c, b, e, dots);
                            cubic_mat += D0 * glm::outerProduct(qa, qb) + dmat3{dot(qa, qb) * D1};
                        }
                    }
                }
                */

                // Linear terms
                const auto lin_mat = dmat3{mu * ed.B(a, c, dots)} + lambda * ed.A(c, a) + mu * ed.A(a, c);

                // const dmat3 total_block = lin_mat + quad_mat + cubic_mat;
                const dmat3 total_block = lin_mat;
                const auto vi_a = 3 * GetVertexIndex(tets, el, a), vi_c = 3 * GetVertexIndex(tets, el, c);
                for (uint k = 0; k < 3; ++k) {
                    for (uint l = 0; l < 3; ++l) {
                        triplets.emplace_back(vi_c + k, vi_a + l, total_block[k][l]);
                    }
                }
            }
        }
    }

    const uint n = 3 * tets.numberofpoints;
    Eigen::SparseMatrix<double> stiffness(n, n);
    stiffness.setFromTriplets(triplets.begin(), triplets.end());
    return stiffness;
}

// Each scalar matrix entry is augmented to a 3x3 diagonal block.
// The resulting matrix is 3*n x 3*n, where n is the number of vertices.
Eigen::SparseMatrix<double> GenerateMassMatrix(const tetgenio &tets, double density) {
    const uint ne = tets.numberoftetrahedra;
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(ne * NEV * NEV * 3);

    // Consistent mass matrix coefficients for a tetrahedron.
    // (See Rao, The Finite Element Method in Engineering, 2004.)
    static constexpr double coeffs[]{
        2, 1, 1, 1,
        1, 2, 1, 1,
        1, 1, 2, 1,
        1, 1, 1, 2
    };

    for (uint el = 0; el < ne; ++el) {
        const double factor = density * GetElementVolume(tets, el) / 20.0;
        for (uint i = 0; i < NEV; i++) {
            const auto vi = GetVertexIndex(tets, el, i);
            const auto row_base = 3 * vi;
            for (uint j = 0; j < NEV; ++j) {
                const auto vj = GetVertexIndex(tets, el, j);
                const auto col_base = 3 * vj;
                const auto entry = factor * coeffs[NEV * j + i];
                // Augment to a 3x3 block (scalar times the identity matrix)
                for (uint k = 0; k < 3; ++k) {
                    triplets.emplace_back(row_base + k, col_base + k, entry);
                }
            }
        }
    }

    const uint n = 3 * tets.numberofpoints;
    Eigen::SparseMatrix<double> mass(n, n);
    mass.setFromTriplets(triplets.begin(), triplets.end());
    return mass;
}

struct ComputeModesOpts {
    float MinModeFreq{20}; // Lowest mode freq
    float MaxModeFreq{16'000}; // Highest mode freq
    std::optional<float> FundamentalFreq{}; // Scale mode freqs so the lowest mode is at this fundamental
    uint NumModes{20}; // Number of synthesized modes
    uint NumFemModes{100}; // Number of modes to be computed with Finite Element Analysis (FEA)
    std::vector<uint> ExPos{}; // Excitation positions
    AcousticMaterialProperties Material{};
};

ModalModes ComputeModes(
    const Eigen::SparseMatrix<double> &M,
    const Eigen::SparseMatrix<double> &K,
    uint num_vertices, uint vertex_dim,
    ComputeModesOpts opts
) {
    /** Compute mass/stiffness eigenvalues and eigenvectors **/
    using OpType = Spectra::SymShiftInvert<double, Eigen::Sparse, Eigen::Sparse>;
    using BOpType = Spectra::SparseSymMatProd<double>;
    using Spectra::GEigsMode::ShiftInvert;

    const uint n = num_vertices * vertex_dim;
    const uint fem_n_modes = std::min(opts.NumFemModes, n - 1);
    const uint convergence_ratio = std::min(std::max(2 * fem_n_modes + 1, 20u), n);
    const double sigma = pow(2 * M_PI * opts.MinModeFreq, 2);
    OpType op{K, M};
    BOpType Bop{M};
    Spectra::SymGEigsShiftSolver<OpType, BOpType, ShiftInvert> eigs{op, Bop, fem_n_modes, convergence_ratio, sigma};
    eigs.init();
    eigs.compute(Spectra::SortRule::LargestMagn, 100, 1e-8, Spectra::SortRule::SmallestAlge);
    if (eigs.info() != Spectra::CompInfo::Successful) return {};

    const auto eigenvalues = eigs.eigenvalues();
    const auto eigenvectors = eigs.eigenvectors();

    /** Compute modes frequencies/gains/T60s **/
    std::vector<float> mode_freqs(fem_n_modes), mode_t60s(fem_n_modes);
    for (uint mode = 0; mode < fem_n_modes; ++mode) {
        if (eigenvalues[mode] < 1) { // Ignore very small eigenvalues
            mode_freqs[mode] = mode_t60s[mode] = 0.0;
            continue;
        }
        // See Eqs. 1-12 in https://www.cs.cornell.edu/~djames/papers/DyRT.pdf for a derivation of the following.
        const double omega_i = sqrt(eigenvalues[mode]); // Undamped natural frequency, in rad/s
        // With good eigenvalue estimates, this should be nearly equivalent:
        // const auto &v = eigenvectors.col(mode);
        // const double Mv = v.transpose() * M * v, Kv = v.transpose() * K * v, omega_i = sqrt(Kv / Mv);
        const double xi_i = 0.5 * (opts.Material.Alpha / omega_i + opts.Material.Beta * omega_i); // Damping ratio
        const double omega_i_hz = omega_i / (2 * M_PI);
        mode_freqs[mode] = omega_i_hz * sqrt(1 - xi_i * xi_i); // Damped natural frequency
        // T60 is the time for the mode's amplitude to decay by 60 dB.
        // 20log10(1000) = 60 dB -> After T60 time, the amplitude is 1/1000th of its initial value.
        // A change of basis gets us to the ln(1000) factor.
        // See https://ccrma.stanford.edu/~jos/st/Audio_Decay_Time_T60.html
        static const double LN_1000 = std::log(1000);
        mode_t60s[mode] = LN_1000 / (xi_i * omega_i_hz); // Damping is based on the _undamped_ natural frequency.
    }

    // Find the lowest mode based on the unscaled minimum frequency.
    // This is because we need to know the lowest mode to scale the other modes.
    const float min_mode_freq = opts.MinModeFreq;
    const uint lowest_mode_i = find_if(mode_freqs | reverse, [min_mode_freq](auto freq) { return freq <= min_mode_freq; }).base() - mode_freqs.begin();
    // Find the highest mode based on the scaled maximum frequency.
    const float max_mode_freq = opts.FundamentalFreq ? mode_freqs[lowest_mode_i] * opts.MaxModeFreq / *opts.FundamentalFreq : opts.MaxModeFreq;
    const uint highest_mode_i = find_if(mode_freqs | drop(lowest_mode_i) | reverse, [max_mode_freq](auto freq) { return freq <= max_mode_freq; }).base() - mode_freqs.begin();

    // Adjust modes to include only the requested range.
    const uint n_modes = std::min(std::min(opts.NumModes, fem_n_modes), highest_mode_i - lowest_mode_i);
    mode_freqs.erase(mode_freqs.begin(), mode_freqs.begin() + lowest_mode_i);
    mode_freqs.resize(n_modes);
    mode_t60s.erase(mode_t60s.begin(), mode_t60s.begin() + lowest_mode_i);
    mode_t60s.resize(n_modes);

    const uint n_ex_pos = std::min(opts.ExPos.size(), size_t(num_vertices));
    std::vector<std::vector<float>> gains(n_ex_pos); // Mode gains by [exitation position][mode]
    for (size_t ex_pos = 0; ex_pos < size_t(n_ex_pos); ++ex_pos) { // For each excitation position
        // If exPos was provided, retrieve data. Otherwise, distribute excitation positions linearly.
        uint ev_i = vertex_dim * (ex_pos < opts.ExPos.size() ? opts.ExPos[ex_pos] : ex_pos * num_vertices / n_ex_pos);
        gains[ex_pos] = std::vector<float>(n_modes);
        float max_gain = 0;
        for (uint mode = 0; mode < n_modes; ++mode) {
            float gain = 0;
            for (uint vi = 0; vi < vertex_dim; ++vi) gain += pow(eigenvectors(ev_i + vi, mode + lowest_mode_i), 2);

            gains[ex_pos][mode] = sqrt(gain);
            if (gains[ex_pos][mode] > max_gain) max_gain = gains[ex_pos][mode];
        }
        for (float &gain : gains[ex_pos]) gain /= max_gain;
    }

    return {std::move(mode_freqs), std::move(mode_t60s), std::move(gains)};
}
} // namespace

ModalModes m2f::mesh2modes(const tetgenio &tets, const AcousticMaterialProperties &material, const std::vector<uint> &excitable_vertices, std::optional<float> fundamental_freq) {
    const auto M = GenerateMassMatrix(tets, material.Density);
    const auto K = ComputeStiffnessMatrix(tets, material);
    return ComputeModes(
        M, K, tets.numberofpoints, 3,
        {
            .MinModeFreq = 20,
            .MaxModeFreq = 16'000,
            .FundamentalFreq = fundamental_freq,
            .NumModes = 30, // number of synthesized modes, starting with the lowest frequency in the provided min/max range
            .NumFemModes = 80, // number of modes to be computed for the finite element analysis
            // Convert to signed ints.
            .ExPos = excitable_vertices,
            .Material = std::move(material),
        }
    );
}
