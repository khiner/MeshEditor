#pragma once

// Computes the mass matrix for the given tet mesh.

#include "sparseMatrix.h"
#include "tetMesh.h"

struct GenerateMassMatrix {
    // Each matrix element z will be augmented to a 3x3 z*I matrix
    // (causing mtx dimensions to grow by a factor of 3).
    // Output matrix will be 3*numVertices x 3*numVertices).
    static void computeMassMatrix(TetMesh *, SparseMatrix **);
};
