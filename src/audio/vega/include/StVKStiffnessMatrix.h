#pragma once

#include "mat3d.h"
#include "sparseMatrix.h"
#include "tetMesh.h"

struct StVKStiffnessMatrix {
    // Compute the tangent stiffness matrix of a StVK elastic deformable object.
    // As a special case, the routine can compute the stiffness matrix in the rest configuration.
    // `vertexDisplacements` is an array of vertex deformations, of length 3*n, where n is the total number of mesh vertices.
    static SparseMatrix ComputeStiffnessMatrix(const TetMesh *, const double *vertexDisplacements);
};
