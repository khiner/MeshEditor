#pragma once

#include "ContactModel.h"
#include "ModalModes.h"

#include <optional>

class tetgenio;
struct AcousticMaterialProperties;

namespace modal {
// Solve parameterization. The eigensolver shift is -(2*pi*MinModeFreq)^2.
struct SolverConfig {
    float MinModeFreq{20}; // Hz
    float MaxModeFreq{16'000}; // Hz
    uint32_t NumModes{30}; // Synthesized modes kept from the FEM eigenpairs
    uint32_t NumFemModes{45}; // Eigenpairs requested from the eigensolver
    double Tolerance{1e-8}; // Eigensolver convergence tolerance
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
};

// FEM modal analysis over quadratic (10-node) tetrahedral elements. Tet geometry is in SI meters,
// so frequencies are in Hz and eigenvectors are mass-normalized. Each excitation position (SI) is
// sampled at its nearest tet point. `baked_scale` (the node's world scale) recovers node-local
// sample positions. ModalModes::Vertices is left empty.
ModalResult mesh2modes(const tetgenio &, const AcousticMaterialProperties &, const std::vector<vec3> &excite_positions, vec3 baked_scale, SolverConfig config = {});
} // namespace modal
