#pragma once

#include "AcousticMaterialProperties.h"
#include "numeric/vec3.h"

#include <cstddef>
#include <vector>

// Stores raw eigenpairs sampled at excitation positions and the material used for the solve.
// modal::RescaleModes re-derives ModalModes for material edits that scale the FEM matrices.
struct ModalEigenSummary {
    std::vector<double> Eigenvalues; // ascending, all solved eigenpairs
    std::vector<std::vector<vec3>> Shapes; // mass-normalized, by [excitation position][eigenpair]
    AcousticMaterialProperties SolvedMaterial{};
    float SolvedMinModeFreq{20}, SolvedMaxModeFreq{16'000}; // The synthesized band the eigenpairs were solved for
    uint32_t SolvedNumModes{30}; // The mode count the eigenpairs were solved for
    size_t TetInputsHash{}; // The tet inputs the eigenpairs were solved over
    std::vector<uint32_t> SolvedVertices;

    bool operator==(const ModalEigenSummary &) const = default;
};
