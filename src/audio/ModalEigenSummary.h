#pragma once

#include "AcousticMaterialProperties.h"
#include "numeric/vec3.h"

#include <cstddef>
#include <vector>

// Stores raw eigenpairs sampled at excitation positions and the material used for the solve.
// modal::RescaleModes re-derives ModalModes for material edits that scale the FEM matrices.
struct ModalEigenSummary : fastfem::ModalEigenSummary {
    float SolvedMinModeFreq{20}, SolvedMaxModeFreq{16'000}; // The synthesized band the eigenpairs were solved for
    uint32_t SolvedNumModes{30}; // The mode count the eigenpairs were solved for
    size_t TetInputsHash{}; // The tet inputs the eigenpairs were solved over
    std::vector<uint32_t> SolvedVertices;

    bool operator==(const ModalEigenSummary &) const = default;
};

namespace zpp::bits {
template<std::size_t>
struct members;
}

auto serialize(const ModalEigenSummary &) -> zpp::bits::members<8>;

constexpr auto serialize(auto &archive, ModalEigenSummary &summary) {
    return archive(summary.Eigenvalues, summary.Shapes, summary.SolvedMaterial, summary.SolvedMinModeFreq, summary.SolvedMaxModeFreq, summary.SolvedNumModes, summary.TetInputsHash, summary.SolvedVertices);
}

constexpr auto serialize(auto &archive, const ModalEigenSummary &summary) {
    return archive(summary.Eigenvalues, summary.Shapes, summary.SolvedMaterial, summary.SolvedMinModeFreq, summary.SolvedMaxModeFreq, summary.SolvedNumModes, summary.TetInputsHash, summary.SolvedVertices);
}
