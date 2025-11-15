#pragma once

#include "numeric/mat4.h"

struct WorldMatrix {
    WorldMatrix(mat4 m) : M{std::move(m)}, MInv{glm::transpose(glm::inverse(M))} {}

    mat4 M; // World-space matrix
    mat4 MInv; // Transpose of inverse
};
