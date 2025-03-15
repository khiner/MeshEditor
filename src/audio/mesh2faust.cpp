#include "mesh2faust.h"

#include "numeric/mat3.h"
#include "numeric/vec3.h"

#include <Eigen/SparseCore>

// Vega
#include "sparseMatrix.h"

// tetgen
#include "tetgen.h"

// Spectra
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/MatOp/SymShiftInvert.h>
#include <Spectra/SymGEigsShiftSolver.h>

#include <memory>
#include <ranges>

using std::ranges::find_if, std::views::drop, std::views::reverse;

namespace {

double GetTetDeterminant(const dvec3 &a, const dvec3 &b, const dvec3 &c, const dvec3 &d) {
    // When det(A) > 0, tet has positive orientation.
    // When det(A) = 0, tet is degenerate.
    // When det(A) < 0, tet has negative orientation.
    return glm::dot(d - a, glm::cross(b - a, c - a));
}
// volume = 1/6 * |(d-a) . ((b-a) x (c-a))|
double GetTetVolume(const dvec3 &a, const dvec3 &b, const dvec3 &c, const dvec3 &d) {
    return (1.f / 6.f) * fabs(GetTetDeterminant(a, b, c, d));
}

int GetVertexIndex(const tetgenio &tets, int element, int vertex) { return tets.tetrahedronlist[element * 4 + vertex]; }

const glm::dvec3 &GetVertex(const tetgenio &tets, int element, int vertex) {
    return reinterpret_cast<const glm::dvec3 &>(tets.pointlist[3 * GetVertexIndex(tets, element, vertex)]);
}

double GetElementVolume(const tetgenio &tets, int el) {
    return GetTetVolume(GetVertex(tets, el, 0), GetVertex(tets, el, 1), GetVertex(tets, el, 2), GetVertex(tets, el, 3));
}

constexpr uint32_t NumTetElementVertices{4};
constexpr uint32_t NEV = NumTetElementVertices; // convenience

using ElementMatrix = double[NEV][NEV];

SparseMatrixOutline GetStiffnessMatrixTopology(const tetgenio &tets) {
    SparseMatrixOutline outline{tets.numberofpoints * 3};
    for (uint32_t el = 0; el < uint32_t(tets.numberoftetrahedra); el++) {
        for (uint32_t i = 0; i < NEV; i++) {
            const auto vi = 3 * GetVertexIndex(tets, el, i);
            for (uint32_t j = 0; j < NEV; j++) {
                const auto vj = 3 * GetVertexIndex(tets, el, j);
                for (uint32_t k = 0; k < 3; k++) {
                    for (uint32_t l = 0; l < 3; l++) {
                        outline.AddEntry(vi + k, vj + l, 0);
                    }
                }
            }
        }
    }
    return outline;
}

struct ElementData {
    double volume;
    dvec3 Phig[NEV]; // gradient of a basis function

    dmat3 A(uint32_t i, uint32_t j) const { return volume * glm::outerProduct(Phig[j], Phig[i]); }
    double B(uint32_t i, uint32_t j, const ElementMatrix &dots) const { return volume * dots[i][j]; }
    dvec3 C(uint32_t i, uint32_t j, uint32_t k, const ElementMatrix &dots) const { return volume * dots[j][k] * Phig[i]; }
    double D(uint32_t i, uint32_t j, uint32_t k, uint32_t l, const ElementMatrix &dots) const { return volume * dots[i][j] * dots[k][l]; }
};

// Create the St.Venant-Kirchhoff A,B,C,D coefficients for a tetrahedral element.
std::vector<ElementData> StVKABCD(const tetgenio &tets) {
    std::vector<ElementData> elements_data(tets.numberoftetrahedra);
    dvec3 columns[2];
    for (uint32_t el = 0; el < elements_data.size(); el++) {
        auto &element_data = elements_data[el];
        // Create the element data structure for a tet
        const double det = GetTetDeterminant(GetVertex(tets, el, 0), GetVertex(tets, el, 1), GetVertex(tets, el, 2), GetVertex(tets, el, 3));
        element_data.volume = fabs(det / 6);
        for (uint32_t i = 0; i < NEV; i++) {
            for (uint32_t j = 0; j < 3; j++) {
                uint32_t ni = 0;
                for (uint32_t ii = 0; ii < NEV; ii++) {
                    if (ii == i) continue;

                    uint32_t nj = 0;
                    for (uint32_t jj = 0; jj < 3; jj++) {
                        if (jj != j) {
                            columns[nj][ni] = GetVertex(tets, el, ii)[jj];
                            nj++;
                        }
                    }
                    const int sign = (i + j) % 2 == 0 ? -1 : 1;
                    element_data.Phig[i][j] = sign * dot(dvec3(1, 1, 1), cross(columns[0], columns[1])) / det;
                    ni++;
                }
            }
        }
    }
    return elements_data;
}

// Compute the tangent stiffness matrix of a StVK elastic deformable object.
// As a special case, the routine can compute the stiffness matrix in the rest configuration.
// `vertexDisplacements` is an array of vertex deformations, of length 3*n, where n is the total number of mesh vertices.
SparseMatrix ComputeStiffnessMatrix(const tetgenio &tets, const AcousticMaterialProperties &material, const double *vertex_displacements) {
    const uint32_t ne = tets.numberoftetrahedra;
    const double lambda = material.Lambda();
    const double mu = material.Mu();
    const auto outline = GetStiffnessMatrixTopology(tets);
    SparseMatrix stiffness{outline};
    std::vector<int> vertices(NEV);
    auto **siffness_data = stiffness.GetDataHandle();
    // Build acceleration indices
    // auto **column_ = (int **)malloc(sizeof(int *) * ne);
    std::vector<std::array<int, NEV * NEV>> column_(ne);
    for (uint32_t el = 0; el < ne; el++) {
        // Seek for value row[j] in list associated with row[i]
        for (uint32_t i = 0; i < NEV; i++) {
            for (uint32_t j = 0; j < NEV; j++) {
                column_[el][NEV * i + j] = stiffness.GetInverseIndex(3 * GetVertexIndex(tets, el, i), 3 * GetVertexIndex(tets, el, j)) / 3;
            }
        }
    }
    dmat3 el_mat;
    auto precomputed_integrals = StVKABCD(tets);
    ElementMatrix dots; // cache dot products
    for (uint32_t el = 0; el < ne; el++) {
        const auto &ed = precomputed_integrals[el];
        for (uint32_t i = 0; i < NEV; i++) {
            for (uint32_t j = 0; j < NEV; j++) {
                dots[i][j] = dot(ed.Phig[i], ed.Phig[j]);
            }
        }

        for (uint32_t v = 0; v < NEV; v++) vertices[v] = GetVertexIndex(tets, el, v);
        for (uint32_t c = 0; c < NEV; c++) {
            const uint32_t rowc = 3 * GetVertexIndex(tets, el, c);
            for (uint32_t a = 0; a < NEV; a++) {
                const auto *qav = &(vertex_displacements[3 * vertices[a]]);
                const dvec3 qa{qav[0], qav[1], qav[2]};
                { // Linear terms
                    el_mat = {mu * ed.B(a, c, dots)}; // diag
                    el_mat += lambda * ed.A(c, a) + mu * ed.A(a, c);
                    // Add a 3x3 block matrix corresponding to a derivative of force on vertex c wrt to vertex a
                    for (uint32_t k = 0; k < 3; k++) {
                        for (uint32_t l = 0; l < 3; l++) {
                            stiffness.AddEntry(rowc + k, 3 * column_[el][NEV * c + a] + l, el_mat[k][l]);
                        }
                    }
                }
                { // Quadratic terms
                    el_mat = {0};
                    for (uint32_t e = 0; e < NEV; e++) {
                        const dvec3 C0 = lambda * ed.C(c, a, e, dots) + mu * (ed.C(e, a, c, dots) + ed.C(a, e, c, dots));
                        const dvec3 C1 = lambda * ed.C(e, a, c, dots) + mu * (ed.C(c, e, a, dots) + ed.C(a, e, c, dots));
                        const dvec3 C2 = lambda * ed.C(a, e, c, dots) + mu * (ed.C(c, a, e, dots) + ed.C(e, a, c, dots));
                        el_mat += glm::outerProduct(C0, qa) + glm::outerProduct(qa, C1) + dmat3{dot(qa, C2)};
                    }
                    // Add matrix block
                    for (uint32_t k = 0; k < 3; k++) {
                        for (uint32_t l = 0; l < 3; l++) {
                            siffness_data[rowc + k][3 * column_[el][NEV * c + a] + l] += el_mat[k][l];
                        }
                    }
                }
                { // Cubic terms
                  // Compute derivative on force on vertex c
                    el_mat = {0};
                    for (uint32_t e = 0; e < NEV; e++) {
                        for (uint32_t b = 0; b < NEV; b++) {
                            const double D0 = lambda * ed.D(a, c, b, e, dots) + mu * (ed.D(a, e, b, c, dots) + ed.D(a, b, c, e, dots));
                            const double D1 = 0.5 * lambda * ed.D(a, b, c, e, dots) + mu * ed.D(a, c, b, e, dots);
                            const auto *qbv = &(vertex_displacements[3 * vertices[b]]);
                            const dvec3 qb{qbv[0], qbv[1], qbv[2]};
                            el_mat += D0 * glm::outerProduct(qa, qb) + dmat3{dot(qa, qb) * D1};
                        }
                    }
                    // Add matrix block
                    for (uint32_t k = 0; k < 3; k++) {
                        for (uint32_t l = 0; l < 3; l++) {
                            siffness_data[rowc + k][3 * column_[el][NEV * c + a] + l] += el_mat[k][l];
                        }
                    }
                }
            }
        }
    }
    return stiffness;
}

// Each matrix element z will be augmented to a 3x3 z*I matrix
// (causing mtx dimensions to grow by a factor of 3).
// Output matrix will be 3*n x 3*n, where n is the number of tet vertices.
SparseMatrix GenerateMassMatrix(const tetgenio &tets, double density) {
    const int n = tets.numberofpoints;
    SparseMatrixOutline outline{3 * n};
    for (int el = 0; el < tets.numberoftetrahedra; el++) {
        /*
        Compute the mass matrix of a single element.
        Consistent mass matrix of a tetrahedron:
                    [ 2  1  1  1  ]
                    [ 1  2  1  1  ]
        mass / 20 * [ 1  1  2  1  ]
                    [ 1  1  1  2  ]
        with mass = density * volume.
        The consistent mass matrix does not depend on the shape of the tetrahedron.
        (Source: Singiresu S. Rao: The finite element method in engineering, 2004)
        */
        static constexpr double coeffs[]{
            2, 1, 1, 1,
            1, 2, 1, 1,
            1, 1, 2, 1,
            1, 1, 1, 2
        };
        const double factor = density * GetElementVolume(tets, el) / 20;
        std::array<double, NEV * NEV> element_mass_matrix;
        for (uint32_t i = 0; i < NEV * NEV; i++) element_mass_matrix[i] = factor * coeffs[i];

        for (uint32_t i = 0; i < NEV; i++) {
            const uint32_t vi = 3 * GetVertexIndex(tets, el, i);
            for (uint32_t j = 0; j < NEV; j++) {
                const uint32_t vj = 3 * GetVertexIndex(tets, el, j);
                const double entry = element_mass_matrix[NEV * j + i];
                outline.AddEntry(vi + 0, vj + 0, entry);
                outline.AddEntry(vi + 1, vj + 1, entry);
                outline.AddEntry(vi + 2, vj + 2, entry);
            }
        }
    }
    return {outline};
}
} // namespace

ModalModes m2f::mesh2modes(const tetgenio &tets, const AcousticMaterialProperties &material, const std::vector<uint32_t> &excitable_vertices, std::optional<float> fundamental_freq) {
    SparseMatrix mass_matrix = GenerateMassMatrix(tets, material.Density);
    const uint32_t vertex_dim = 3;
    const uint32_t num_vertices = tets.numberofpoints;
    // In linear modal analysis, the displacements are zero.
    double *displacements = (double *)calloc(num_vertices * vertex_dim, sizeof(double));
    const auto stiffness_matrix = ComputeStiffnessMatrix(tets, material, displacements);
    free(displacements);

    // Copy Vega sparse matrices to Eigen matrices.
    // _Note: Eigen is column-major by default._
    std::vector<Eigen::Triplet<double, uint32_t>> K_triplets, M_triplets;
    for (uint32_t i = 0; i < uint32_t(stiffness_matrix.GetNumRows()); ++i) {
        for (uint32_t j = 0; j < uint32_t(stiffness_matrix.GetRowLength(i)); ++j) {
            K_triplets.push_back({i, uint32_t(stiffness_matrix.GetColumnIndex(i, j)), stiffness_matrix.GetEntry(i, j)});
        }
    }
    for (uint32_t i = 0; i < uint32_t(mass_matrix.GetNumRows()); ++i) {
        for (uint32_t j = 0; j < uint32_t(mass_matrix.GetRowLength(i)); ++j) {
            M_triplets.push_back({i, uint32_t(mass_matrix.GetColumnIndex(i, j)), mass_matrix.GetEntry(i, j)});
        }
    }

    const uint32_t n = stiffness_matrix.Getn();
    Eigen::SparseMatrix<double> K(n, n), M(n, n);
    K.setFromTriplets(K_triplets.begin(), K_triplets.end());
    M.setFromTriplets(M_triplets.begin(), M_triplets.end());

    return mesh2modes(
        M, K, num_vertices, vertex_dim,
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

ModalModes m2f::mesh2modes(
    const Eigen::SparseMatrix<double> &M,
    const Eigen::SparseMatrix<double> &K,
    uint32_t num_vertices, uint32_t vertex_dim,
    Args args
) {
    const uint32_t fem_n_modes = std::min(args.NumFemModes, num_vertices * vertex_dim - 1);

    /** Compute mass/stiffness eigenvalues and eigenvectors **/
    using OpType = Spectra::SymShiftInvert<double, Eigen::Sparse, Eigen::Sparse>;
    using BOpType = Spectra::SparseSymMatProd<double>;

    const uint32_t convergence_ratio = std::min(std::max(2u * fem_n_modes + 1u, 20u), num_vertices * vertex_dim);
    const double sigma = pow(2 * M_PI * args.MinModeFreq, 2);
    OpType op(K, M);
    BOpType Bop(M);
    Spectra::SymGEigsShiftSolver<OpType, BOpType, Spectra::GEigsMode::ShiftInvert> eigs(op, Bop, fem_n_modes, convergence_ratio, sigma);
    eigs.init();
    eigs.compute(Spectra::SortRule::LargestMagn, 1000, 1e-10, Spectra::SortRule::SmallestAlge);
    if (eigs.info() != Spectra::CompInfo::Successful) return {};

    const auto eigenvalues = eigs.eigenvalues();
    const auto eigenvectors = eigs.eigenvectors();

    // Compute modes frequencies/gains/T60s
    std::vector<float> mode_freqs(fem_n_modes), mode_t60s(fem_n_modes);
    for (uint32_t mode = 0; mode < fem_n_modes; ++mode) {
        if (eigenvalues[mode] < 1) { // Ignore very small eigenvalues
            mode_freqs[mode] = mode_t60s[mode] = 0.0;
            continue;
        }
        // See Eqs. 1-12 in https://www.cs.cornell.edu/~djames/papers/DyRT.pdf for a derivation of the following.
        const double omega_i = sqrt(eigenvalues[mode]); // Undamped natural frequency, in rad/s
        // With good eigenvalue estimates, this should be nearly equivalent:
        // const auto &v = eigenvectors.col(mode);
        // const double Mv = v.transpose() * M * v, Kv = v.transpose() * K * v, omega_i = sqrt(Kv / Mv);
        const double xi_i = 0.5 * (args.Material.Alpha / omega_i + args.Material.Beta * omega_i); // Damping ratio
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
    const float min_mode_freq = args.MinModeFreq;
    const uint32_t lowest_mode_i = find_if(mode_freqs | reverse, [min_mode_freq](auto freq) { return freq <= min_mode_freq; }).base() - mode_freqs.begin();
    // Find the highest mode based on the scaled maximum frequency.
    const float max_mode_freq = args.FundamentalFreq ? mode_freqs[lowest_mode_i] * args.MaxModeFreq / *args.FundamentalFreq : args.MaxModeFreq;
    const uint32_t highest_mode_i = find_if(mode_freqs | drop(lowest_mode_i) | reverse, [max_mode_freq](auto freq) { return freq <= max_mode_freq; }).base() - mode_freqs.begin();

    // Adjust modes to include only the requested range.
    const uint32_t n_modes = std::min(std::min(args.NumModes, fem_n_modes), highest_mode_i - lowest_mode_i);
    mode_freqs.erase(mode_freqs.begin(), mode_freqs.begin() + lowest_mode_i);
    mode_freqs.resize(n_modes);
    mode_t60s.erase(mode_t60s.begin(), mode_t60s.begin() + lowest_mode_i);
    mode_t60s.resize(n_modes);

    const uint32_t n_ex_pos = std::min(args.ExPos.size(), size_t(num_vertices));
    std::vector<std::vector<float>> gains(n_ex_pos); // Mode gains by [exitation position][mode]
    for (size_t ex_pos = 0; ex_pos < size_t(n_ex_pos); ++ex_pos) { // For each excitation position
        // If exPos was provided, retrieve data. Otherwise, distribute excitation positions linearly.
        uint32_t ev_i = vertex_dim * (ex_pos < args.ExPos.size() ? args.ExPos[ex_pos] : ex_pos * num_vertices / n_ex_pos);
        gains[ex_pos] = std::vector<float>(n_modes);
        float max_gain = 0;
        for (uint32_t mode = 0; mode < n_modes; ++mode) {
            float gain = 0;
            for (uint32_t vi = 0; vi < vertex_dim; ++vi) gain += pow(eigenvectors(ev_i + vi, mode + lowest_mode_i), 2);

            gains[ex_pos][mode] = sqrt(gain);
            if (gains[ex_pos][mode] > max_gain) max_gain = gains[ex_pos][mode];
        }
        for (float &gain : gains[ex_pos]) gain /= max_gain;
    }

    return {std::move(mode_freqs), std::move(mode_t60s), std::move(gains)};
}
