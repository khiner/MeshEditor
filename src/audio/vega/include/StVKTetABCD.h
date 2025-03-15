#pragma once

#include "mat3d.h"
#include "tetMesh.h"

// Stores the St.Venant-Kirchhoff A,B,C,D coefficients for a tetrahedral element.
struct StVKTetABCD {
    StVKTetABCD(TetMesh *tetMesh) {
        int numElements = tetMesh->getNumElements();
        elementsData = (ElementData *)malloc(sizeof(ElementData) * numElements);
        for (int el = 0; el < numElements; el++) {
            Vec3d vertices[4];
            for (int i = 0; i < 4; i++) vertices[i] = tetMesh->getVertex(el, i);
            StVKSingleTetABCD(vertices, &elementsData[el]);
        }
    }
    ~StVKTetABCD() {
        free(elementsData);
    }

    struct ElementData {
        double volume;
        Vec3d Phig[4]; // gradient of a basis function
    };

    struct ElementCache {
        ElementData *element;
        double dots[4][4];

        Mat3d A(int i, int j) { return element->volume * tensorProduct(element->Phig[i], element->Phig[j]); }
        double B(int i, int j) { return element->volume * dots[i][j]; }
        Vec3d C(int i, int j, int k) { return element->volume * dots[j][k] * element->Phig[i]; }
        double D(int i, int j, int k, int l) { return element->volume * dots[i][j] * dots[k][l]; }
    };

    void PrepareElement(int el, ElementCache *cache) {
        cache->element = &elementsData[el];
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                cache->dots[i][j] = dot(cache->element->Phig[i], cache->element->Phig[j]);
    }

private:
    ElementData *elementsData;

    // creates the ElementData structure for a tet
    void StVKSingleTetABCD(Vec3d vtx[4], ElementData *target) {
        double det = TetMesh::getTetDeterminant(vtx[0], vtx[1], vtx[2], vtx[3]);
        target->volume = fabs(det / 6);

        for (int i = 0; i < TetMesh::NumElementVertices; i++)
            for (int j = 0; j < 3; j++) {
                Vec3d columns[2];
                int countI = 0;
                for (int ii = 0; ii < TetMesh::NumElementVertices; ii++) {
                    if (ii == i) continue;

                    int countJ = 0;
                    for (int jj = 0; jj < 3; jj++) {
                        if (jj == j) continue;

                        columns[countJ][countI] = vtx[ii][jj];
                        countJ++;
                    }
                    int sign = ((i + j) % 2) == 0 ? -1 : 1;
                    target->Phig[i][j] = sign * dot(Vec3d(1, 1, 1), cross(columns[0], columns[1])) / det;
                    countI++;
                }
            }
    }
};
