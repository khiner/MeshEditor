#pragma once

#include "ModalModes.h"

#include <optional>
#include <vector>

class tetgenio;
struct AcousticMaterialProperties;

namespace m2f {
ModalModes mesh2modes(const tetgenio &, const AcousticMaterialProperties &, const std::vector<uint32_t> &excitable_vertices, std::optional<float> fundamental_freq);
} // namespace m2f
