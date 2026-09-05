#pragma once

#include "AcousticMaterialProperties.h"
#include "numeric/vec3.h"

#include <cstddef>
#include <vector>

// Stores raw eigenpairs sampled at excitation positions and the material used for the solve.
// modal::RescaleModes re-derives ModalModes for material edits that scale the FEM matrices.
struct ModalEigenSummary : fastfem::ModalEigenSummary {
    size_t OperatorHash{};
    size_t ModalConfigHash{};
    std::vector<uint32_t> SolvedVertices;

    bool operator==(const ModalEigenSummary &) const = default;
};

namespace zpp::bits {
template<std::size_t>
struct members;
}

auto serialize(const ModalEigenSummary &) -> zpp::bits::members<6>;

constexpr auto serialize(auto &archive, ModalEigenSummary &summary) {
    return archive(summary.Eigenvalues, summary.Shapes, summary.SolvedMaterial, summary.OperatorHash, summary.ModalConfigHash, summary.SolvedVertices);
}

constexpr auto serialize(auto &archive, const ModalEigenSummary &summary) {
    return archive(summary.Eigenvalues, summary.Shapes, summary.SolvedMaterial, summary.OperatorHash, summary.ModalConfigHash, summary.SolvedVertices);
}
