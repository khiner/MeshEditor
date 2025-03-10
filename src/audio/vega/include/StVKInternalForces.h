#pragma once

#include "StVKTetABCD.h"
#include "tetMesh.h"

struct StVKInternalForces {
    StVKInternalForces(TetMesh *, StVKTetABCD *, bool addGravity = false, double g = 9.81);
    ~StVKInternalForces();

    // both vertex displacements and internal forces refer to the vertices of the simulation mesh
    // they must be (pre-allocated) vectors of length 3 * numVertices
    // the internal forces are returned with the sign corresponding to f_int(x) on the left side of the equation M * x'' + f_int(x) = f_ext
    // i.e., the computed internal forces are negatives of the actual physical internal forces acting on the material
    void ComputeForces(const double *vertexDisplacements, double *internalForces);

    // enables or disables the gravity (note: you can also set this in the constructor; use this routine to turn the gravity on/off during the simulation)
    void SetGravity(bool addGravity) {
        this->addGravity = addGravity;
        InitGravity();
    } // if addGravity is enabled, ComputeForces will subtract the gravity force from the internal forces (note: subtraction, not addition, is used because the internal forces are returned with the sign as described in the f_int(x) comment above)

    TetMesh *GetTetMesh() { return tetMesh; }
    StVKTetABCD *GetPrecomputedIntegrals() { return precomputedIntegrals; }

    void AddLinearTermsContribution(const double *vertexDisplacements, double *forces, int elementLow = -1, int elementHigh = -1);
    void AddQuadraticTermsContribution(const double *vertexDisplacements, double *forces, int elementLow = -1, int elementHigh = -1);
    void AddCubicTermsContribution(const double *vertexDisplacements, double *forces, int elementLow = -1, int elementHigh = -1);

private:
    TetMesh *tetMesh;
    StVKTetABCD *precomputedIntegrals;

    double *gravityForce;
    bool addGravity;
    double g;
    void InitGravity(); // aux function

    double *buffer;
    int numElementVertices;

    double *lambdaLame;
    double *muLame;
};
