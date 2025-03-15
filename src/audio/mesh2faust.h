#pragma once

#include "AcousticMaterialProperties.h"
#include "ModalModes.h"

#include <optional>
#include <string>
#include <string_view>
#include <vector>

class tetgenio;

namespace Eigen {
template<typename _Scalar, int _Flags, typename _StorageIndex>
class SparseMatrix;
} // namespace Eigen

namespace m2f {
struct Args {
    float MinModeFreq = 20; // Lowest mode freq
    float MaxModeFreq = 16'000; // Highest mode freq
    std::optional<float> FundamentalFreq = std::nullopt; // Scale mode freqs so the lowest mode is at this fundamental
    uint32_t NumModes = 20; // Number of synthesized modes
    uint32_t NumFemModes = 100; // Number of modes to be computed with Finite Element Analysis (FEA)
    std::vector<uint32_t> ExPos = {}; // Excitation positions
    AcousticMaterialProperties Material = {};
};

ModalModes mesh2modes(const tetgenio &, const AcousticMaterialProperties &, const std::vector<uint32_t> &excitable_vertices, std::optional<float> fundamental_freq);

ModalModes mesh2modes(
    const Eigen::SparseMatrix<double, 0, int> &M, // Mass matrix
    const Eigen::SparseMatrix<double, 0, int> &K, // Stiffness matrix
    uint32_t num_vertices,
    uint32_t vertex_dim = 3,
    Args args = {}
);
} // namespace m2f
