#pragma once

#include "ModalModes.h"

#include <optional>

class tetgenio;
struct AcousticMaterialProperties;

namespace m2f {
// Sample positions are divided by `scale` to recover node-local coordinates.
ModalModes mesh2modes(const tetgenio &, const AcousticMaterialProperties &, const std::vector<uint32_t> &excitable_vertices, vec3 scale, std::optional<float> fundamental_freq);
} // namespace m2f
