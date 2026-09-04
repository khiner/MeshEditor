#pragma once

#include <FastFEM/Surface2Modes.h>

#include <cstddef>

// Stores the latest full eigenvector basis for modal::SolveReuse with matching tetrahedral inputs.
// One scene-level slot supports the object under active editing.
struct ModalWarmStart {
    size_t TetInputsHash{};
    fastfem::ModeBasis Basis{};
};
