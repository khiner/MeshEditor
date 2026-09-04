#pragma once

#include "ContactModel.h"
#include "ModalEigenSummary.h"
#include "ModalModes.h"
#include "mesh/TetMeshData.h"

#include <FastFEM/Surface2Modes.h>

#include <expected>
#include <optional>
#include <span>
#include <string>

namespace modal {
struct SurfaceSolveResult {
    ModalModes Modes;
    MassProperties Mass;
    ModalEigenSummary Summary;
    TetMeshData Tetrahedra;
    fastfem::ModeBasis Basis;
    std::vector<uint32_t> SamplePointOfExcitation;
};

std::expected<SurfaceSolveResult, std::string> SolveSurfaceModes(
    std::span<const vec3> positions, std::span<const uint32_t> triangle_indices,
    const AcousticMaterialProperties &, std::span<const vec3> excitation_positions,
    vec3 baked_scale, fastfem::Discretization, fastfem::SurfaceSolveConfig = {},
    fastfem::SolveReuse = {}, fastfem::SolveMonitor * = nullptr
);

std::optional<ModalModes> RescaleModes(
    const ModalEigenSummary &, const ModalModes &current,
    const AcousticMaterialProperties &, fastfem::SolverConfig = {}
);
} // namespace modal
