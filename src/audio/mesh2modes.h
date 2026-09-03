#pragma once

#include "ContactModel.h"
#include "ModalEigenSummary.h"
#include "ModalModes.h"

#include <Eigen/Core>

#include <optional>
#include <span>

struct TetMesh;
struct JobMonitor;

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
    // The index into Modes.Positions of each requested excitation position, in request order.
    // Requests reaching the same tet point share one entry there.
    std::vector<uint32_t> SamplePointOfExcitation;
};

// An eigenvector basis from identical tetrahedral inputs initializes block iteration with WarmTolerance.
struct SolveReuse {
    const Eigen::MatrixXf *SeedBasis{};
    bool KeepBasis{}; // Fill ModalResult::Basis
};

ModalResult mesh2modes(const TetMesh &, const AcousticMaterialProperties &, const std::vector<vec3> &excite_positions, vec3 baked_scale, SolverConfig config = {}, SolveReuse reuse = {}, JobMonitor *monitor = nullptr);

ModalModes PostprocessModes(std::span<const double> eigenvalues, const std::vector<std::vector<vec3>> &shapes, float shape_scale, const AcousticMaterialProperties &, const SolverConfig &, std::vector<vec3> positions);

// Re-derives a modal model after Young's modulus or density changes with fixed tetrahedral inputs.
// FEM eigenvalues scale by (E'/E)/(rho'/rho).
// Mass-normalized shapes scale by 1/sqrt(rho'/rho), while vertices, positions, and baked scale remain unchanged.
// from `current`. Empty when the edit is not exactly scalable (Poisson ratio differs).
std::optional<ModalModes> RescaleModes(const ModalEigenSummary &, const ModalModes &current, const AcousticMaterialProperties &, SolverConfig config = {});
} // namespace modal
