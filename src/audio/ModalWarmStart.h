#pragma once

#include <Eigen/Core>

#include <cstddef>
#include <memory>

// Stores the latest full eigenvector basis for modal::SolveReuse with matching tetrahedral inputs.
// One scene-level slot supports the object under active editing.
struct ModalWarmStart {
    size_t TetInputsHash{};
    std::shared_ptr<const Eigen::MatrixXf> Basis{};
};
