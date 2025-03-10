#include "tetMesh.h"

#include <cmath>

void TetMesh::computeElementMassMatrix(int el, double *massMatrix) const {
    /*
      Consistent mass matrix of a tetrahedron =
                     [ 2  1  1  1  ]
                     [ 1  2  1  1  ]
         mass / 20 * [ 1  1  2  1  ]
                     [ 1  1  1  2  ]

      Note: mass = density * volume. Other than via the mass, the
      consistent mass matrix does not depend on the shape of the tetrahedron.
      (This can be seen after a long algebraic derivation; see:
       Singiresu S. Rao: The finite element method in engineering, 2004)
    */

    static constexpr double mtx[16] = {2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2};
    double factor =  getElementDensity(el) * getElementVolume(el) / 20;
    for (int i = 0; i < 16; i++) massMatrix[i] = factor * mtx[i];
}

double TetMesh::getTetVolume(const Vec3d &a, const Vec3d &b, const Vec3d &c, const Vec3d &d) {
    // volume = 1/6 * | (d-a) . ((b-a) x (c-a)) |
    return (1.0 / 6 * fabs(getTetDeterminant(a, b, c, d)));
}

double TetMesh::getElementVolume(int el) const {
    return getTetVolume(getVertex(el, 0), getVertex(el, 1), getVertex(el, 2), getVertex(el, 3));
}

double TetMesh::getTetDeterminant(const Vec3d &a, const Vec3d &b, const Vec3d &c, const Vec3d &d) {
    // computes det(A), for the 4x4 matrix A
    //     [ 1 a ]
    // A = [ 1 b ]
    //     [ 1 c ]
    //     [ 1 d ]
    // It can be shown that det(A) = dot(d - a, cross(b - a, c - a))
    // When det(A) > 0, the tet has positive orientation.
    // When det(A) = 0, the tet is degenerate.
    // When det(A) < 0, the tet has negative orientation.

    return dot(d - a, cross(b - a, c - a));
}
