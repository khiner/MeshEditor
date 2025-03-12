#include "StVKStiffnessMatrix.h"

static constexpr auto NEV = TetMesh::NumElementVertices;

StVKStiffnessMatrix::StVKStiffnessMatrix(StVKInternalForces *stVKInternalForces) {
    precomputedIntegrals = stVKInternalForces->GetPrecomputedIntegrals();
    tetMesh = stVKInternalForces->GetTetMesh();
    int numElements = tetMesh->getNumElements();

    lambdaLame = (double *)malloc(sizeof(double) * numElements);
    muLame = (double *)malloc(sizeof(double) * numElements);

    for (int el = 0; el < numElements; el++) {
        const auto &material = tetMesh->material;
        lambdaLame[el] = material.getLambda();
        muLame[el] = material.getMu();
    }

    SparseMatrix topology = GetStiffnessMatrixTopology();
    // build acceleration indices
    row_ = (int **)malloc(sizeof(int *) * numElements);
    column_ = (int **)malloc(sizeof(int *) * numElements);
    for (int el = 0; el < numElements; el++) {
        row_[el] = (int *)malloc(sizeof(int) * NEV);
        column_[el] = (int *)malloc(sizeof(int) * NEV * NEV);
        for (uint32_t ver = 0; ver < NEV; ver++) row_[el][ver] = tetMesh->getVertexIndex(el, ver);
        // seek for value row[j] in list associated with row[i]
        for (uint32_t i = 0; i < NEV; i++)
            for (uint32_t j = 0; j < NEV; j++)
                column_[el][NEV * i + j] = topology.GetInverseIndex(3 * row_[el][i], 3 * row_[el][j]) / 3;
    }
}

SparseMatrix StVKStiffnessMatrix::GetStiffnessMatrixTopology() {
    std::array<int, NEV> vertices;
    SparseMatrixOutline outline{3 * tetMesh->getNumVertices()};
    for (int el = 0; el < tetMesh->getNumElements(); el++) {
        for (uint32_t ver = 0; ver < NEV; ver++) vertices[ver] = tetMesh->getVertexIndex(el, ver);

        for (uint32_t i = 0; i < NEV; i++) {
            for (uint32_t j = 0; j < NEV; j++) {
                for (uint32_t k = 0; k < 3; k++) {
                    for (uint32_t l = 0; l < 3; l++) {
                        // Only add the entry if both vertices are free (non-fixed).
                        // The corresponding elt is in row 3*i+k, column 3*j+l
                        outline.AddEntry(3 * vertices[i] + k, 3 * vertices[j] + l, 0.0);
                    }
                }
            }
        }
    }

    return {&outline};
}

StVKStiffnessMatrix::~StVKStiffnessMatrix() {
    int numElements = tetMesh->getNumElements();
    for (int i = 0; i < numElements; i++) free(row_[i]);
    free(row_);

    for (int i = 0; i < numElements; i++) free(column_[i]);
    free(column_);

    free(lambdaLame);
    free(muLame);
}

// the master function
void StVKStiffnessMatrix::ComputeStiffnessMatrix(const double *vertexDisplacements, SparseMatrix *sparseMatrix) {
    sparseMatrix->ResetToZero();
    AddLinearTermsContribution(vertexDisplacements, sparseMatrix);
    AddQuadraticTermsContribution(vertexDisplacements, sparseMatrix);
    AddCubicTermsContribution(vertexDisplacements, sparseMatrix);
}

void StVKStiffnessMatrix::AddLinearTermsContribution(const double *vertexDisplacements, SparseMatrix *sparseMatrix, int elementLow, int elementHigh) {
    if (elementLow < 0) elementLow = 0;
    if (elementHigh < 0) elementHigh = tetMesh->getNumElements();

    int *vertices = (int *)malloc(sizeof(int) * NEV);
    void *elIter;
    precomputedIntegrals->AllocateElementIterator(&elIter);

    for (int el = elementLow; el < elementHigh; el++) {
        precomputedIntegrals->PrepareElement(el, elIter);
        for (uint32_t ver = 0; ver < NEV; ver++) vertices[ver] = tetMesh->getVertexIndex(el, ver);

        double lambda = lambdaLame[el];
        double mu = muLame[el];
        // over all vertices of the voxel, computing row of vertex c
        for (uint32_t c = 0; c < NEV; c++) {
            // linear terms, over all vertices
            for (uint32_t a = 0; a < NEV; a++) {
                Mat3d matrix(1.0);
                matrix *= mu * precomputedIntegrals->B(elIter, a, c);
                matrix += lambda * precomputedIntegrals->A(elIter, c, a) + mu * precomputedIntegrals->A(elIter, a, c);
                AddMatrix3x3Block(c, a, el, matrix, sparseMatrix);
            }
        }
    }

    free(vertices);
    precomputedIntegrals->ReleaseElementIterator(elIter);
}

#define ADD_MATRIX_BLOCK(where)                                                      \
    for (k = 0; k < 3; k++)                                                          \
        for (l = 0; l < 3; l++) {                                                    \
            dataHandle[rowc + k][3 * column[c8 + (where)] + l] += matrix[3 * k + l]; \
        }

void StVKStiffnessMatrix::AddQuadraticTermsContribution(const double *vertexDisplacements, SparseMatrix *sparseMatrix, int elementLow, int elementHigh) {
    if (elementLow < 0) elementLow = 0;
    if (elementHigh < 0) elementHigh = tetMesh->getNumElements();

    int *vertices = (int *)malloc(sizeof(int) * NEV);
    void *elIter;
    precomputedIntegrals->AllocateElementIterator(&elIter);

    double **dataHandle = sparseMatrix->GetDataHandle();
    for (int el = elementLow; el < elementHigh; el++) {
        precomputedIntegrals->PrepareElement(el, elIter);
        int *row = row_[el];
        int *column = column_[el];

        for (uint32_t ver = 0; ver < NEV; ver++) vertices[ver] = tetMesh->getVertexIndex(el, ver);

        double lambda = lambdaLame[el];
        double mu = muLame[el];
        // over all vertices of the voxel, computing row of vertex c
        for (uint32_t c = 0; c < NEV; c++) {
            int rowc = 3 * row[c];
            int c8 = NEV * c;
            // quadratic terms, compute contribution to block (c,e) of the stiffness matrix
            for (uint32_t e = 0; e < NEV; e++) {
                double matrix[9];
                memset(matrix, 0, sizeof(double) * 9);
                for (uint32_t a = 0; a < NEV; a++) {
                    double qa[3] = {vertexDisplacements[3 * vertices[a] + 0], vertexDisplacements[3 * vertices[a] + 1], vertexDisplacements[3 * vertices[a] + 2]};

                    Vec3d C0v = lambda * precomputedIntegrals->C(elIter, c, a, e) + mu * (precomputedIntegrals->C(elIter, e, a, c) + precomputedIntegrals->C(elIter, a, e, c));
                    double C0[3] = {C0v[0], C0v[1], C0v[2]};

                    // C0 tensor qa
                    matrix[0] += C0[0] * qa[0];
                    matrix[1] += C0[0] * qa[1];
                    matrix[2] += C0[0] * qa[2];
                    matrix[3] += C0[1] * qa[0];
                    matrix[4] += C0[1] * qa[1];
                    matrix[5] += C0[1] * qa[2];
                    matrix[6] += C0[2] * qa[0];
                    matrix[7] += C0[2] * qa[1];
                    matrix[8] += C0[2] * qa[2];

                    Vec3d C1v = lambda * precomputedIntegrals->C(elIter, e, a, c) + mu * (precomputedIntegrals->C(elIter, c, e, a) + precomputedIntegrals->C(elIter, a, e, c));
                    double C1[3] = {C1v[0], C1v[1], C1v[2]};

                    // qa tensor C1
                    matrix[0] += qa[0] * C1[0];
                    matrix[1] += qa[0] * C1[1];
                    matrix[2] += qa[0] * C1[2];
                    matrix[3] += qa[1] * C1[0];
                    matrix[4] += qa[1] * C1[1];
                    matrix[5] += qa[1] * C1[2];
                    matrix[6] += qa[2] * C1[0];
                    matrix[7] += qa[2] * C1[1];
                    matrix[8] += qa[2] * C1[2];

                    Vec3d C2v = lambda * precomputedIntegrals->C(elIter, a, e, c) + mu * (precomputedIntegrals->C(elIter, c, a, e) + precomputedIntegrals->C(elIter, e, a, c));
                    double C2[3] = {C2v[0], C2v[1], C2v[2]};

                    // qa dot C2
                    double dotp = qa[0] * C2[0] + qa[1] * C2[1] + qa[2] * C2[2];
                    matrix[0] += dotp;
                    matrix[4] += dotp;
                    matrix[8] += dotp;
                }
                int k, l;
                ADD_MATRIX_BLOCK(e);
            }
        }
    }

    free(vertices);
    precomputedIntegrals->ReleaseElementIterator(elIter);
}

void StVKStiffnessMatrix::AddCubicTermsContribution(const double *vertexDisplacements, SparseMatrix *sparseMatrix, int elementLow, int elementHigh) {
    if (elementLow < 0) elementLow = 0;
    if (elementHigh < 0) elementHigh = tetMesh->getNumElements();

    int *vertices = (int *)malloc(sizeof(int) * NEV);
    void *elIter;
    precomputedIntegrals->AllocateElementIterator(&elIter);

    double **dataHandle = sparseMatrix->GetDataHandle();
    for (int el = elementLow; el < elementHigh; el++) {
        precomputedIntegrals->PrepareElement(el, elIter);
        int *row = row_[el];
        int *column = column_[el];

        for (uint32_t ver = 0; ver < NEV; ver++) vertices[ver] = tetMesh->getVertexIndex(el, ver);

        double lambda = lambdaLame[el];
        double mu = muLame[el];

        // over all vertices of the voxel, computing derivative on force on vertex c
        for (uint32_t c = 0; c < NEV; c++) {
            int rowc = 3 * row[c];
            int c8 = NEV * c;
            // cubic terms, compute contribution to block (c,e) of the stiffness matrix
            for (uint32_t e = 0; e < NEV; e++) {
                double matrix[9];
                memset(matrix, 0, sizeof(double) * 9);
                for (uint32_t a = 0; a < NEV; a++) {
                    int va = vertices[a];
                    const double *qa = &(vertexDisplacements[3 * va]);
                    for (uint32_t b = 0; b < NEV; b++) {
                        int vb = vertices[b];
                        const double *qb = &(vertexDisplacements[3 * vb]);
                        double D0 = lambda * precomputedIntegrals->D(elIter, a, c, b, e) +
                            mu * (precomputedIntegrals->D(elIter, a, e, b, c) + precomputedIntegrals->D(elIter, a, b, c, e));

                        matrix[0] += D0 * qa[0] * qb[0];
                        matrix[1] += D0 * qa[0] * qb[1];
                        matrix[2] += D0 * qa[0] * qb[2];
                        matrix[3] += D0 * qa[1] * qb[0];
                        matrix[4] += D0 * qa[1] * qb[1];
                        matrix[5] += D0 * qa[1] * qb[2];
                        matrix[6] += D0 * qa[2] * qb[0];
                        matrix[7] += D0 * qa[2] * qb[1];
                        matrix[8] += D0 * qa[2] * qb[2];

                        double D1 = 0.5 * lambda * precomputedIntegrals->D(elIter, a, b, c, e) +
                            mu * precomputedIntegrals->D(elIter, a, c, b, e);
                        double dotpD = D1 * (qa[0] * qb[0] + qa[1] * qb[1] + qa[2] * qb[2]);
                        matrix[0] += dotpD;
                        matrix[4] += dotpD;
                        matrix[8] += dotpD;
                    }
                }
                int k, l;
                ADD_MATRIX_BLOCK(e);
            }
        }
    }

    free(vertices);
    precomputedIntegrals->ReleaseElementIterator(elIter);
}
