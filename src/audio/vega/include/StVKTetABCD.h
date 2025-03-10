#pragma once

#include "mat3d.h"
#include "tetMesh.h"

// Stores the St.Venant-Kirchhoff A,B,C,D coefficients for a tetrahedral element.
struct StVKTetABCD {
    StVKTetABCD(TetMesh *);
    ~StVKTetABCD();

    Mat3d A(void *elementIterator, int i, int j);
    double B(void *elementIterator, int i, int j);
    Vec3d C(void *elementIterator, int i, int j, int k);
    double D(void *elementIterator, int i, int j, int k, int l);

    typedef struct
    {
        double volume;
        Vec3d Phig[4]; // gradient of a basis function
    } elementData;

    typedef struct
    {
        elementData *elementPointer;
        double dots[4][4];
    } ElementCache;

    void AllocateElementIterator(void **elementIterator) {
        ElementCache *cache = new ElementCache();
        *elementIterator = cache;
    }

    void ReleaseElementIterator(void *elementIterator) {
        ElementCache *cache = (ElementCache *)elementIterator;
        delete (cache);
    }

    void PrepareElement(int el, void *elementIterator) {
        ElementCache *cache = (ElementCache *)elementIterator;
        cache->elementPointer = &elementsData[el];
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                (cache->dots)[i][j] = dot(cache->elementPointer->Phig[i], cache->elementPointer->Phig[j]);
    }

private:
    elementData *elementsData;

    // creates the elementData structure for a tet
    void StVKSingleTetABCD(Vec3d vertices[4], elementData *target);
};
