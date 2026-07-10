#pragma once

#include "ContactModel.h"
#include "ModalModes.h"

#include <optional>

class tetgenio;
struct AcousticMaterialProperties;

namespace modal {
struct ModalResult {
    ModalModes Modes;
    MassProperties MassProps;
};

// FEM modal analysis over quadratic (10-node) tetrahedral elements. Tet geometry is in SI meters,
// so frequencies are in Hz and eigenvectors are mass-normalized. Each excitation position (SI) is
// sampled at its nearest tet point. `baked_scale` (the node's world scale) recovers node-local
// sample positions. ModalModes::Vertices is left empty.
ModalResult mesh2modes(const tetgenio &, const AcousticMaterialProperties &, const std::vector<vec3> &excite_positions, vec3 baked_scale, std::optional<float> fundamental_freq);
} // namespace modal
