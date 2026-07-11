#pragma once

#include "ContactModel.h"
#include "ModalEigenSummary.h"
#include "ModalModes.h"

#include <Eigen/Core>

#include <optional>
#include <span>

class tetgenio;

namespace modal {
// Solve parameterization. The eigensolver shift is -(2*pi*MinModeFreq)^2.
struct SolverConfig {
    float MinModeFreq{20}; // Hz
    float MaxModeFreq{16'000}; // Hz
    uint32_t NumModes{30}; // Synthesized modes kept from the FEM eigenpairs
    uint32_t NumFemModes{45}; // Eigenpairs requested from the eigensolver
    double Tolerance{1e-8}; // Eigensolver convergence tolerance
    double WarmTolerance{1e-4}; // Warm-started re-solve tolerance (relative eigenvalue change between block iterations)
    uint32_t MaxRestarts{100}; // Eigensolver restart limit
    std::optional<float> FundamentalFreq{}; // Scale mode freqs so the lowest mode is at this fundamental
};

// Wall-clock seconds per solve stage, with problem-size counters.
// OpSolve is the shift-inverted linear solves, a subset of Iterate.
struct SolveProfile {
    double MassProps{}, QuadMesh{}, Assemble{}, SampleExcite{};
    double Factorize{}, Iterate{}, OpSolve{}, Extract{};
    uint32_t Dofs{}, StiffnessNonZeros{}, OpApplications{}, Restarts{};

    SolveProfile &operator+=(const SolveProfile &o) {
        MassProps += o.MassProps;
        QuadMesh += o.QuadMesh;
        Assemble += o.Assemble;
        SampleExcite += o.SampleExcite;
        Factorize += o.Factorize;
        Iterate += o.Iterate;
        OpSolve += o.OpSolve;
        Extract += o.Extract;
        Dofs += o.Dofs;
        StiffnessNonZeros += o.StiffnessNonZeros;
        OpApplications += o.OpApplications;
        Restarts += o.Restarts;
        return *this;
    }
};

struct ModalResult {
    ModalModes Modes;
    MassProperties MassProps;
    SolveProfile Profile;
    ModalEigenSummary Summary; // Raw eigenpairs at the excitation positions (TetInputsHash left 0)
    Eigen::MatrixXf Basis; // Full eigenvector basis, filled when SolveReuse::KeepBasis
};

// A prior solve's eigenvector basis over the same tet inputs seeds the eigensolver, which
// re-converges it in a few block iterations (WarmTolerance) instead of solving from scratch.
struct SolveReuse {
    const Eigen::MatrixXf *SeedBasis{};
    bool KeepBasis{}; // Fill ModalResult::Basis
};

// FEM modal analysis over quadratic (10-node) tetrahedral elements. Tet geometry is in SI meters,
// so frequencies are in Hz and eigenvectors are mass-normalized. Each excitation position (SI) is
// sampled at its nearest tet point. ModalModes::Vertices is left empty.
// `baked_scale` (the node's world scale) recovers node-local sample positions.
ModalResult mesh2modes(const tetgenio &, const AcousticMaterialProperties &, const std::vector<vec3> &excite_positions, vec3 baked_scale, SolverConfig config = {}, SolveReuse reuse = {});

// Mode frequencies, T60s, and shapes from raw eigenpairs: filter to the audible window, apply
// damping and optional fundamental scaling. `shapes` holds each excitation position's mode-shape
// vector per eigenpair, scaled by `shape_scale` into the result. ModalModes::Vertices is left empty.
ModalModes PostprocessModes(std::span<const double> eigenvalues, const std::vector<std::vector<vec3>> &shapes, float shape_scale, const AcousticMaterialProperties &, const SolverConfig &, std::vector<vec3> positions);

// Exact re-derivation of the modal model under a material edit at unchanged tet inputs: Young's
// modulus and density scale the FEM matrices linearly, so eigenvalues scale by (E'/E)/(rho'/rho)
// and mass-normalized shapes by 1/sqrt(rho'/rho). Vertices, positions, and baked scale carry over
// from `current`. Empty when the edit is not exactly scalable (Poisson ratio differs).
std::optional<ModalModes> RescaleModes(const ModalEigenSummary &, const ModalModes &current, const AcousticMaterialProperties &, SolverConfig config = {});
} // namespace modal
