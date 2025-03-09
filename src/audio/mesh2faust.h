#pragma once

#include "AcousticMaterialProperties.h"
#include "ModalModes.h"

#include <optional>
#include <string>
#include <vector>

class TetMesh;
class tetgenio;

namespace Eigen {
template<typename _Scalar, int _Flags, typename _StorageIndex>
class SparseMatrix;
} // namespace Eigen

namespace m2f {
struct CommonArguments {
    float ModesMinFreq = 20; // Lowest mode freq
    float ModesMaxFreq = 10000; // Highest mode freq
    uint32_t TargetNModes = 20; // Number of synthesized modes
    uint32_t FemNModes = 100; // Number of modes to be computed with Finite Element Analysis (FEA)
    std::vector<int> ExPos = {}; // Specific excitation positions
    std::optional<int> NExPos = {}; // Number of excitation positions (default is max)
};

ModalModes mesh2modal(TetMesh *, AcousticMaterialProperties material = {}, CommonArguments args = {});
ModalModes mesh2modal(const tetgenio &, const AcousticMaterialProperties &, const std::vector<uint32_t> &excitable_vertices, std::optional<float> fundamental_freq);

ModalModes mesh2modal(
    const Eigen::SparseMatrix<double, 0, int> &M, // Mass matrix
    const Eigen::SparseMatrix<double, 0, int> &K, // Stiffness matrix
    uint32_t num_vertices,
    uint32_t vertex_dim = 3,
    AcousticMaterialProperties material = {},
    CommonArguments args = {}
);

// Subset of `CommonArguments` required for the DSP code generation phase
struct DspGenArguments {
    std::string modelName = "modalModel"; // Name for the generated model
    bool freqControl = false; // Freq control activated
};

// Generate DSP code from modal modes.
std::string modal2faust(const ModalModes &, DspGenArguments args = {});

} // namespace m2f
