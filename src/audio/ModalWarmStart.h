#pragma once

#include <Eigen/Core>

#include <cstddef>
#include <memory>

// Registry-ctx slot holding the most recent modal solve's full eigenvector basis, seeding the
// next solve over matching tet inputs (see modal::SolveReuse). One app-wide slot: it accelerates
// re-solving the object under active edit, and is dropped with the scene.
struct ModalWarmStart {
    size_t TetInputsHash{};
    std::shared_ptr<const Eigen::MatrixXf> Basis{};
};
