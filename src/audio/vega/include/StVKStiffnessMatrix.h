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
    void GetStiffnessMatrixTopology(SparseMatrix **);

    // evaluates the tangent stiffness matrix in the given deformation configuration
    // "vertexDisplacements" is an array of vertex deformations, of length 3*n, where n is the total number of mesh vertices
    void ComputeStiffnessMatrix(const double *vertexDisplacements, SparseMatrix *);

    TetMesh *GetTetMesh() { return tetMesh; }
    StVKTetABCD *GetPrecomputedIntegrals() { return precomputedIntegrals; }

    // === the routines below are meant for advanced usage ===

    // auxiliary functions, these will add the contributions into 'forces'
    void AddLinearTermsContribution(const double *vertexDisplacements, SparseMatrix *, int elementLow = -1, int elementHigh = -1);
    void AddQuadraticTermsContribution(const double *vertexDisplacements, SparseMatrix *, int elementLow = -1, int elementHigh = -1);
    void AddCubicTermsContribution(const double *vertexDisplacements, SparseMatrix *, int elementLow = -1, int elementHigh = -1);

private:
    int numElementVertices;

    // acceleration indices
    int **row_;
    int **column_;

    TetMesh *tetMesh;
    StVKInternalForces *stVKInternalForces;
    StVKTetABCD *precomputedIntegrals;

    double *lambdaLame;
    double *muLame;

    // adds a 3x3 block matrix corresponding to a derivative of force on vertex c wrt to vertex a
    // c is 0..7
    // a is 0..7
    void AddMatrix3x3Block(int c, int a, int element, Mat3d &matrix, SparseMatrix *sparseMatrix) {
        int *row = row_[element];
        int *column = column_[element];

        for (int k = 0; k < 3; k++)
            for (int l = 0; l < 3; l++)
                sparseMatrix->AddEntry(3 * row[c] + k, 3 * column[numElementVertices * c + a] + l, matrix[k][l]);
    }
};
