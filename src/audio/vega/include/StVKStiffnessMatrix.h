#pragma once

/*
  Computes the tangent stiffness matrix of a StVK elastic deformable object.
  The tangent stiffness matrix depends on the deformable configuration.
  As a special case, the routine can compute the stiffness matrix in the rest configuration.
*/

#include "StVKInternalForces.h"
#include "mat3d.h"
#include "sparseMatrix.h"

struct StVKStiffnessMatrix {
    StVKStiffnessMatrix(StVKInternalForces *);
    ~StVKStiffnessMatrix();

    // generates a zero matrix with the same pattern of non-zero entries as the tangent stiffness matrix
    // note: sparsity pattern does not depend on the deformable configuration
    SparseMatrix GetStiffnessMatrixTopology();

    // evaluates the tangent stiffness matrix in the given deformation configuration
    // "vertexDisplacements" is an array of vertex deformations, of length 3*n, where n is the total number of mesh vertices
    void ComputeStiffnessMatrix(const double *vertexDisplacements, SparseMatrix *);

private:
    // acceleration indices
    int **row_;
    int **column_;

    TetMesh *tetMesh;
    StVKInternalForces *stVKInternalForces;
    StVKTetABCD *precomputedIntegrals;

    double *lambdaLame;
    double *muLame;
};
