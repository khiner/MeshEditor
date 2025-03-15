#include "StVKStiffnessMatrix.h"

static constexpr auto NEV = TetMesh::NumElementVertices;

StVKStiffnessMatrix::StVKStiffnessMatrix(TetMesh *tetMesh, StVKTetABCD *precomputedABCDIntegrals)
    : precomputedIntegrals(precomputedABCDIntegrals), tetMesh(tetMesh) {
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
        for (uint32_t i = 0; i < NEV; i++) {
            for (uint32_t j = 0; j < NEV; j++) {
                column_[el][NEV * i + j] = topology.GetInverseIndex(3 * row_[el][i], 3 * row_[el][j]) / 3;
            }
        }
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
                        outline.AddEntry(3 * vertices[i] + k, 3 * vertices[j] + l, 0);
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
    int *vertices = (int *)malloc(sizeof(int) * NEV);
    auto *elCache = new StVKTetABCD::ElementCache();
    double **dataHandle = sparseMatrix->GetDataHandle();
    for (int el = 0; el < tetMesh->getNumElements(); el++) {
        precomputedIntegrals->PrepareElement(el, elCache);
        for (uint32_t ver = 0; ver < NEV; ver++) vertices[ver] = tetMesh->getVertexIndex(el, ver);

        int *row = row_[el];
        int *column = column_[el];
        double lambda = lambdaLame[el];
        double mu = muLame[el];
        for (uint32_t c = 0; c < NEV; c++) {
            uint32_t rowc = 3 * row[c];
            uint32_t c8 = NEV * c;
            double matrix[9];
            for (uint32_t a = 0; a < NEV; a++) {
                const Vec3d qa{&(vertexDisplacements[3 * vertices[a]])};
                { // Linear terms
                    Mat3d matrix(1.0);
                    matrix *= mu * elCache->B(a, c);
                    matrix += lambda * elCache->A(c, a) + mu * elCache->A(a, c);
                    // Add a 3x3 block matrix corresponding to a derivative of force on vertex c wrt to vertex a
                    for (int k = 0; k < 3; k++) {
                        for (int l = 0; l < 3; l++) {
                            sparseMatrix->AddEntry(3 * row[c] + k, 3 * column[NEV * c + a] + l, matrix[k][l]);
                        }
                    }
                }
                { // Quadratic terms
                    memset(matrix, 0, sizeof(double) * 9);
                    for (uint32_t e = 0; e < NEV; e++) {
                        const Vec3d C0 = lambda * elCache->C(c, a, e) + mu * (elCache->C(e, a, c) + elCache->C(a, e, c));
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

                        const Vec3d C1 = lambda * elCache->C(e, a, c) + mu * (elCache->C(c, e, a) + elCache->C(a, e, c));
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

                        const Vec3d C2 = lambda * elCache->C(a, e, c) + mu * (elCache->C(c, a, e) + elCache->C(e, a, c));
                        // qa dot C2
                        const double dotp = dot(qa, C2);
                        matrix[0] += dotp;
                        matrix[4] += dotp;
                        matrix[8] += dotp;
                    }
                    // Add matrix block
                    for (uint32_t k = 0; k < 3; k++) {
                        for (uint32_t l = 0; l < 3; l++) {
                            dataHandle[rowc + k][3 * column[c8 + a] + l] += matrix[3 * k + l];
                        }
                    }
                }
                { // Cubic terms
                  // Compute derivative on force on vertex c
                    memset(matrix, 0, sizeof(double) * 9);
                    for (uint32_t e = 0; e < NEV; e++) {
                        for (uint32_t b = 0; b < NEV; b++) {
                            const Vec3d qb{&(vertexDisplacements[3 * vertices[b]])};
                            const double D0 = lambda * elCache->D(a, c, b, e) + mu * (elCache->D(a, e, b, c) + elCache->D(a, b, c, e));
                            matrix[0] += D0 * qa[0] * qb[0];
                            matrix[1] += D0 * qa[0] * qb[1];
                            matrix[2] += D0 * qa[0] * qb[2];
                            matrix[3] += D0 * qa[1] * qb[0];
                            matrix[4] += D0 * qa[1] * qb[1];
                            matrix[5] += D0 * qa[1] * qb[2];
                            matrix[6] += D0 * qa[2] * qb[0];
                            matrix[7] += D0 * qa[2] * qb[1];
                            matrix[8] += D0 * qa[2] * qb[2];

                            const double D1 = 0.5 * lambda * elCache->D(a, b, c, e) + mu * elCache->D(a, c, b, e);
                            const double dotpD = D1 * dot(qa, qb);
                            matrix[0] += dotpD;
                            matrix[4] += dotpD;
                            matrix[8] += dotpD;
                        }
                    }
                    // Add matrix block
                    for (uint32_t k = 0; k < 3; k++) {
                        for (uint32_t l = 0; l < 3; l++) {
                            dataHandle[rowc + k][3 * column[c8 + a] + l] += matrix[3 * k + l];
                        }
                    }
                }
            }
        }
    }
    free(vertices);
    delete elCache;
}
