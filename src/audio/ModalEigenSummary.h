#pragma once

#include "AcousticMaterialProperties.h"
#include "numeric/vec3.h"

#include <cstddef>
#include <vector>

// A modal solve's raw eigenpairs sampled at the excitation positions, kept on the sound entity.
// With the solved material, this exactly re-derives ModalModes under material edits that scale
// the FEM matrices (see modal::RescaleModes), so those edits need no solve.
struct ModalEigenSummary {
    std::vector<double> Eigenvalues; // ascending, all solved eigenpairs
    std::vector<std::vector<vec3>> Shapes; // mass-normalized, by [excitation position][eigenpair]
    AcousticMaterialProperties SolvedMaterial{};
    float SolvedMinModeFreq{20}, SolvedMaxModeFreq{16'000}; // The synthesized band the eigenpairs were solved for
    uint32_t SolvedNumModes{30}; // The mode count the eigenpairs were solved for
    size_t TetInputsHash{}; // The tet inputs the eigenpairs were solved over

    bool operator==(const ModalEigenSummary &) const = default;
};
