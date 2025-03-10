#pragma once

// A loader class for StVK cubic (unreduced) coefficients.

#include "StVKElementABCDLoader.h"
#include "StVKTetABCD.h"
#include "tetMesh.h"

struct StVKElementABCDLoader {
    inline static StVKTetABCD *load(TetMesh *mesh) { return new StVKTetABCD(mesh); }
};
