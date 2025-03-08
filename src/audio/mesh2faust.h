#pragma once

#include "AcousticMaterialProperties.h"

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
    int TargetNModes = 20; // Number of synthesized modes
    int FemNModes = 100; // Number of modes to be computed with Finite Element Analysis (FEA)
    std::vector<int> ExPos = {}; // Specific excitation positions
    std::optional<int> NExPos = {}; // Number of excitation positions (default is max)
};

struct ModalModel {
    std::vector<float> ModeFreqs; // Mode frequencies
    std::vector<float> ModeT60s; // Mode T60 decay times
    std::vector<std::vector<float>> ModeGains; // Mode gains by [exitation position][mode]
};

ModalModel mesh2modal(TetMesh *, AcousticMaterialProperties material = {}, CommonArguments args = {});
ModalModel mesh2modal(const tetgenio &, const AcousticMaterialProperties &, const std::vector<uint32_t> &excitable_vertices, std::optional<float> fundamental_freq);

ModalModel mesh2modal(
    const Eigen::SparseMatrix<double, 0, int> &M, // Mass matrix
    const Eigen::SparseMatrix<double, 0, int> &K, // Stiffness matrix
    int num_vertices,
    int vertex_dim = 3,
    AcousticMaterialProperties material = {},
    CommonArguments args = {}
);

// Subset of `CommonArguments` required for the DSP code generation phase
struct DspGenArguments {
    std::string modelName = "modalModel"; // Name for the generated model
    bool freqControl = false; // Freq control activated
};

// Generate DSP code from a `ModalModel`.
std::string modal2faust(const ModalModel &, DspGenArguments args = {});

} // namespace m2f
