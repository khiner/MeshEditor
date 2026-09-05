#pragma once

#include <FastFEM/Surface2Modes.h>

#include <cstddef>

// Stores the latest full eigenvector basis for matching Tet10 operator inputs.
// One scene-level slot supports the object under active editing.
struct ModalWarmStart {
    size_t OperatorHash{};
    fastfem::ModeBasis Basis{};
};
