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
    int *vertices = (int *)malloc(sizeof(int) * NEV);
    auto *elCache = new StVKTetABCD::ElementCache();
    for (int el = 0; el < tetMesh->getNumElements(); el++) {
        precomputedIntegrals->PrepareElement(el, elCache);
        for (uint32_t ver = 0; ver < NEV; ver++) vertices[ver] = tetMesh->getVertexIndex(el, ver);
        double lambda = lambdaLame[el];
        double mu = muLame[el];
        for (uint32_t c = 0; c < NEV; c++) {
            for (uint32_t a = 0; a < NEV; a++) {
                const Vec3d qa{&(vertexDisplacements[3 * vertices[a]])};
                { // Linear terms
                    const Vec3d force = lambda * elCache->A(c, a) * qa + mu * elCache->B(a, c) * qa + mu * elCache->A(a, c) * qa;
                    forces[3 * vertices[c] + 0] += force[0];
                    forces[3 * vertices[c] + 1] += force[1];
                    forces[3 * vertices[c] + 2] += force[2];
                }
                { // Quadratic terms
                    for (uint32_t b = 0; b < NEV; b++) {
                        const Vec3d qb{&(vertexDisplacements[3 * vertices[b]])};
                        const double dotp = dot(qa, qb);
                        const Vec3d forceTerm1 = 0.5 * lambda * dotp * elCache->C(c, a, b) + mu * dotp * elCache->C(a, b, c);
                        const Vec3d C = lambda * elCache->C(a, b, c) + mu * (elCache->C(c, a, b) + elCache->C(b, a, c));
                        const double dotCqa = dot(C, qa);
                        forces[3 * vertices[c] + 0] += forceTerm1[0] + dotCqa * qb[0];
                        forces[3 * vertices[c] + 1] += forceTerm1[1] + dotCqa * qb[1];
                        forces[3 * vertices[c] + 2] += forceTerm1[2] + dotCqa * qb[2];
                    }
                }
                { // Cubic terms
                    for (uint32_t b = 0; b < NEV; b++) {
                        const Vec3d qb{&(vertexDisplacements[3 * vertices[b]])};
                        const double dotp = dot(qa, qb);
                        for (uint32_t d = 0; d < NEV; d++) {
                            const Vec3d qd{&(vertexDisplacements[3 * vertices[d]])};
                            double scalar = dotp * (0.5 * lambda * elCache->D(a, b, c, d) + mu * elCache->D(a, c, b, d));
                            double *force = &(forces[3 * vertices[c]]);
                            force[0] += scalar * qd[0];
                            force[1] += scalar * qd[1];
                            force[2] += scalar * qd[2];
                        }
                    }
                }
            }
        }
    }
    free(vertices);
    delete elCache;

    if (addGravity) {
        uint32_t n = tetMesh->getNumVertices();
        for (uint32_t i = 0; i < 3 * n; i++) forces[i] -= gravityForce[i];
    }
}
