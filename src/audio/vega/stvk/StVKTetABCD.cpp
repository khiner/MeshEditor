#include "StVKTetABCD.h"

#include <cmath>

StVKTetABCD::StVKTetABCD(TetMesh *tetMesh) {
    int numElements = tetMesh->getNumElements();
    elementsData = (elementData *)malloc(sizeof(elementData) * numElements);
    for (int el = 0; el < numElements; el++) {
        Vec3d vertices[4];
        for (int i = 0; i < 4; i++) vertices[i] = tetMesh->getVertex(el, i);
        StVKSingleTetABCD(vertices, &elementsData[el]);
    }
}

StVKTetABCD::~StVKTetABCD() {
    free(elementsData);
}

void StVKTetABCD::StVKSingleTetABCD(Vec3d vtx[4], elementData *target) {
    double det = TetMesh::getTetDeterminant(vtx[0], vtx[1], vtx[2], vtx[3]);
    target->volume = fabs(det / 6);

    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 3; j++) {
            Vec3d columns[2];
            int countI = 0;
            for (int ii = 0; ii < 4; ii++) {
                if (ii == i)
                    continue;
                int countJ = 0;
                for (int jj = 0; jj < 3; jj++) {
                    if (jj == j)
                        continue;

                    columns[countJ][countI] = vtx[ii][jj];
                    countJ++;
                }
                int sign = (((i + j) % 2) == 0) ? -1 : 1;
                target->Phig[i][j] = 1.0 * sign * dot(Vec3d(1, 1, 1), cross(columns[0], columns[1])) / det;
                countI++;
            }
        }
}

Mat3d StVKTetABCD::A(void *elementIterator, int i, int j) {
    ElementCache *cache = (ElementCache *)elementIterator;
    return cache->elementPointer->volume * tensorProduct(cache->elementPointer->Phig[i], cache->elementPointer->Phig[j]);
}

double StVKTetABCD::B(void *elementIterator, int i, int j) {
    ElementCache *cache = (ElementCache *)elementIterator;
    return cache->elementPointer->volume * cache->dots[i][j];
}

Vec3d StVKTetABCD::C(void *elementIterator, int i, int j, int k) {
    ElementCache *cache = (ElementCache *)elementIterator;
    return cache->elementPointer->volume * cache->dots[j][k] * cache->elementPointer->Phig[i];
}

double StVKTetABCD::D(void *elementIterator, int i, int j, int k, int l) {
    ElementCache *cache = (ElementCache *)elementIterator;
    return cache->elementPointer->volume * cache->dots[i][j] * cache->dots[k][l];
}
