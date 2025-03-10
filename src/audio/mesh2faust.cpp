#include "mesh2faust.h"

#include <Eigen/SparseCore>

// Vega
#include "StVKElementABCDLoader.h"
#include "StVKStiffnessMatrix.h"
#include "generateMassMatrix.h"
#include "tetMesh.h"

// tetgen
#include "tetgen.h"

// Spectra
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/MatOp/SymShiftInvert.h>
#include <Spectra/SymGEigsShiftSolver.h>

#include <ranges>

using std::ranges::find_if, std::views::drop, std::views::reverse;

ModalModes m2f::mesh2modes(TetMesh &tet_mesh, Args args) {
    SparseMatrix *mass_matrix;
    GenerateMassMatrix::computeMassMatrix(&tet_mesh, &mass_matrix);

    StVKTetABCD *precomputed_integrals = StVKElementABCDLoader::load(&tet_mesh);
    StVKInternalForces internal_forces{&tet_mesh, precomputed_integrals};
    SparseMatrix *stiffness_matrix;
    StVKStiffnessMatrix stiffness_matrix_class{&internal_forces};
    stiffness_matrix_class.GetStiffnessMatrixTopology(&stiffness_matrix);

    const uint32_t vertex_dim = 3;
    const uint32_t num_vertices = tet_mesh.getNumVertices();
    // In linear modal analysis, the displacements are zero.
    double *displacements = (double *)calloc(num_vertices * vertex_dim, sizeof(double));
    stiffness_matrix_class.ComputeStiffnessMatrix(displacements, stiffness_matrix);

    free(displacements);
    delete precomputed_integrals;
    precomputed_integrals = nullptr;

    // Copy Vega sparse matrices to Eigen matrices.
    // _Note: Eigen is column-major by default._
    std::vector<Eigen::Triplet<double, uint32_t>> K_triplets, M_triplets;
    for (uint32_t i = 0; i < uint32_t(stiffness_matrix->GetNumRows()); ++i) {
        for (uint32_t j = 0; j < uint32_t(stiffness_matrix->GetRowLength(i)); ++j) {
            K_triplets.push_back({i, uint32_t(stiffness_matrix->GetColumnIndex(i, j)), stiffness_matrix->GetEntry(i, j)});
        }
    }
    for (uint32_t i = 0; i < uint32_t(mass_matrix->GetNumRows()); ++i) {
        for (uint32_t j = 0; j < uint32_t(mass_matrix->GetRowLength(i)); ++j) {
            M_triplets.push_back({i, uint32_t(mass_matrix->GetColumnIndex(i, j)), mass_matrix->GetEntry(i, j)});
        }
    }

    const uint32_t n = stiffness_matrix->Getn();
    Eigen::SparseMatrix<double> K(n, n), M(n, n);
    K.setFromTriplets(K_triplets.begin(), K_triplets.end());
    M.setFromTriplets(M_triplets.begin(), M_triplets.end());

    delete mass_matrix;
    mass_matrix = nullptr;
    delete stiffness_matrix;
    stiffness_matrix = nullptr;

    return mesh2modes(M, K, num_vertices, vertex_dim, std::move(args));
}

ModalModes m2f::mesh2modes(const tetgenio &tets, const AcousticMaterialProperties &material, const std::vector<uint32_t> &excitable_vertices, std::optional<float> fundamental_freq) {
    // Convert the tetrahedral mesh into a VegaFEM TetMesh.
    std::vector<int> tet_indices;
    tet_indices.reserve(tets.numberoftetrahedra * 4 * 3); // 4 triangles per tetrahedron, 3 indices per triangle.
    // Turn each tetrahedron into 4 triangles.
    for (uint32_t i = 0; i < uint32_t(tets.numberoftetrahedra); ++i) {
        auto &indices = tets.tetrahedronlist;
        uint32_t tri_i = i * 4;
        int a = indices[tri_i], b = indices[tri_i + 1], c = indices[tri_i + 2], d = indices[tri_i + 3];
        tet_indices.insert(tet_indices.end(), {a, b, c, d, a, b, c, d, a, b, c, d});
    }
    TetMesh tet_mesh{
        tets.numberofpoints, tets.pointlist, tets.numberoftetrahedra * 3, tet_indices.data(),
        material.YoungModulus, material.PoissonRatio, material.Density
    };

    return mesh2modes(
        tet_mesh,
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
