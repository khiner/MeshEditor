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

// Sample positions are divided by `scale` (the tet scale) to recover node-local coordinates. `baked_scale` (the node's
// world scale) converts those node-local lengths to SI for the mass properties.
ModalResult mesh2modes(const tetgenio &, const AcousticMaterialProperties &, const std::vector<uint32_t> &excitable_vertices, vec3 scale, vec3 baked_scale, std::optional<float> fundamental_freq);
} // namespace modal
