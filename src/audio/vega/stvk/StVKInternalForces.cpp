#include "StVKInternalForces.h"

static constexpr auto NEV = TetMesh::NumElementVertices;

StVKInternalForces::StVKInternalForces(TetMesh *tetMesh, StVKTetABCD *precomputedABCDIntegrals_, bool addGravity_, double g_)
    : tetMesh(tetMesh), precomputedIntegrals(precomputedABCDIntegrals_), gravityForce(NULL), addGravity(addGravity_), g(g_) {
    int numElements = tetMesh->getNumElements();
    lambdaLame = (double *)malloc(sizeof(double) * numElements);
    muLame = (double *)malloc(sizeof(double) * numElements);

    for (int el = 0; el < numElements; el++) {
        const auto &material = tetMesh->material;
        lambdaLame[el] = material.getLambda();
        muLame[el] = material.getMu();
    }

    buffer = (double *)malloc(sizeof(double) * 3 * tetMesh->getNumVertices());
    InitGravity();
}

StVKInternalForces::~StVKInternalForces() {
    free(gravityForce);
    free(buffer);
    free(lambdaLame);
    free(muLame);
}

void StVKInternalForces::InitGravity() {
    if (addGravity && (gravityForce == NULL)) {
        gravityForce = (double *)malloc(sizeof(double) * 3 * tetMesh->getNumVertices());
        tetMesh->computeGravity(gravityForce, g);
    }
}

void StVKInternalForces::ComputeForces(const double *vertexDisplacements, double *forces) {
    memset(forces, 0, sizeof(double) * 3 * tetMesh->getNumVertices());
    AddLinearTermsContribution(vertexDisplacements, forces);
    AddQuadraticTermsContribution(vertexDisplacements, forces);
    AddCubicTermsContribution(vertexDisplacements, forces);

    if (addGravity) {
        int n = tetMesh->getNumVertices();
        for (int i = 0; i < 3 * n; i++) forces[i] -= gravityForce[i];
    }
}

void StVKInternalForces::AddLinearTermsContribution(const double *vertexDisplacements, double *forces, int elementLow, int elementHigh) {
    if (elementLow < 0) elementLow = 0;
    if (elementHigh < 0) elementHigh = tetMesh->getNumElements();

    int *vertices = (int *)malloc(sizeof(int) * NEV);
    auto *elCache = new StVKTetABCD::ElementCache();
    for (int el = elementLow; el < elementHigh; el++) {
        precomputedIntegrals->PrepareElement(el, elCache);
        for (uint32_t ver = 0; ver < NEV; ver++) vertices[ver] = tetMesh->getVertexIndex(el, ver);

        double lambda = lambdaLame[el];
        double mu = muLame[el];
        // over all vertices of the voxel, computing force on vertex c
        for (uint32_t c = 0; c < NEV; c++) {
            // linear terms, over all vertices
            for (uint32_t a = 0; a < NEV; a++) {
                Vec3d qa(vertexDisplacements[3 * vertices[a] + 0], vertexDisplacements[3 * vertices[a] + 1], vertexDisplacements[3 * vertices[a] + 2]);
                Vec3d force = lambda * (elCache->A(c, a) * qa) + mu * elCache->B(a, c) * qa + mu * elCache->A(a, c) * qa;
                forces[3 * vertices[c] + 0] += force[0];
                forces[3 * vertices[c] + 1] += force[1];
                forces[3 * vertices[c] + 2] += force[2];
            }
        }
    }

    free(vertices);
    delete elCache;
}

void StVKInternalForces::AddQuadraticTermsContribution(const double *vertexDisplacements, double *forces, int elementLow, int elementHigh) {
    if (elementLow < 0) elementLow = 0;
    if (elementHigh < 0) elementHigh = tetMesh->getNumElements();

    int *vertices = (int *)malloc(sizeof(int) * NEV);
    auto *elCache = new StVKTetABCD::ElementCache();
    for (int el = elementLow; el < elementHigh; el++) {
        precomputedIntegrals->PrepareElement(el, elCache);
        for (uint32_t ver = 0; ver < NEV; ver++) vertices[ver] = tetMesh->getVertexIndex(el, ver);

        double lambda = lambdaLame[el];
        double mu = muLame[el];
        // over all vertices of the voxel, computing force on vertex c
        for (uint32_t c = 0; c < NEV; c++) {
            // quadratic terms, over all vertices
            for (uint32_t a = 0; a < NEV; a++)
                for (uint32_t b = 0; b < NEV; b++) {
                    double qa[3] = {vertexDisplacements[3 * vertices[a] + 0], vertexDisplacements[3 * vertices[a] + 1], vertexDisplacements[3 * vertices[a] + 2]};
                    double qb[3] = {vertexDisplacements[3 * vertices[b] + 0], vertexDisplacements[3 * vertices[b] + 1], vertexDisplacements[3 * vertices[b] + 2]};
                    double dotp = qa[0] * qb[0] + qa[1] * qb[1] + qa[2] * qb[2];
                    Vec3d forceTerm1 = 0.5 * lambda * dotp * elCache->C(c, a, b) + mu * dotp * elCache->C(a, b, c);
                    Vec3d C = lambda * elCache->C(a, b, c) + mu * (elCache->C(c, a, b) + elCache->C(b, a, c));
                    double dotCqa = C[0] * qa[0] + C[1] * qa[1] + C[2] * qa[2];
                    forces[3 * vertices[c] + 0] += forceTerm1[0] + dotCqa * qb[0];
                    forces[3 * vertices[c] + 1] += forceTerm1[1] + dotCqa * qb[1];
                    forces[3 * vertices[c] + 2] += forceTerm1[2] + dotCqa * qb[2];
                }
        }
    }

    free(vertices);
    delete elCache;
}

void StVKInternalForces::AddCubicTermsContribution(const double *vertexDisplacements, double *forces, int elementLow, int elementHigh) {
    if (elementLow < 0) elementLow = 0;
    if (elementHigh < 0) elementHigh = tetMesh->getNumElements();

    int *vertices = (int *)malloc(sizeof(int) * NEV);
    auto *elCache = new StVKTetABCD::ElementCache();
    for (int el = elementLow; el < elementHigh; el++) {
        precomputedIntegrals->PrepareElement(el, elCache);
        for (uint32_t ver = 0; ver < NEV; ver++) vertices[ver] = tetMesh->getVertexIndex(el, ver);

        double lambda = lambdaLame[el];
        double mu = muLame[el];
        // over all vertices of the voxel, computing force on vertex c
        for (uint32_t c = 0; c < NEV; c++) {
            int vc = vertices[c];
            // cubic terms, over all vertices
            for (uint32_t a = 0; a < NEV; a++) {
                int va = vertices[a];
                for (uint32_t b = 0; b < NEV; b++) {
                    int vb = vertices[b];
                    for (uint32_t d = 0; d < NEV; d++) {
                        int vd = vertices[d];
                        const double *qa = &(vertexDisplacements[3 * va]);
                        const double *qb = &(vertexDisplacements[3 * vb]);
                        const double *qd = &(vertexDisplacements[3 * vd]);
                        double dotp = qa[0] * qb[0] + qa[1] * qb[1] + qa[2] * qb[2];
                        double scalar = dotp * (0.5 * lambda * elCache->D(a, b, c, d) + mu * elCache->D(a, c, b, d));
                        double *force = &(forces[3 * vc]);
                        force[0] += scalar * qd[0];
                        force[1] += scalar * qd[1];
                        force[2] += scalar * qd[2];
                    }
                }
            }
        }
    }

    free(vertices);
    delete elCache;
}
