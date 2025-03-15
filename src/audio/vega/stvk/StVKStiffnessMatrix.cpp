#include "StVKStiffnessMatrix.h"

#include "numeric/mat3.h"

#include <vector>

namespace {
constexpr auto NEV = TetMesh::NumElementVertices;

using ElementMatrix = double[NEV][NEV];

SparseMatrixOutline GetStiffnessMatrixTopology(const TetMesh &tets) {
    const uint32_t ne = tets.getNumElements();
    SparseMatrixOutline outline{3 * tets.getNumVertices()};
    for (uint32_t el = 0; el < ne; el++) {
        for (uint32_t i = 0; i < NEV; i++) {
            for (uint32_t j = 0; j < NEV; j++) {
                for (uint32_t k = 0; k < 3; k++) {
                    for (uint32_t l = 0; l < 3; l++) {
                        // Only add the entry if both vertices are free (non-fixed).
                        // The corresponding element is in row 3*i+k, column 3*j+l
                        outline.AddEntry(3 * tets.getVertexIndex(el, i) + k, 3 * tets.getVertexIndex(el, j) + l, 0);
                    }
                }
            }
        }
    }
    return outline;
}

struct ElementData {
    double volume;
    dvec3 Phig[NEV]; // gradient of a basis function

    dmat3 A(uint32_t i, uint32_t j) const { return volume * glm::outerProduct(Phig[j], Phig[i]); }
    double B(uint32_t i, uint32_t j, const ElementMatrix &dots) const { return volume * dots[i][j]; }
    dvec3 C(uint32_t i, uint32_t j, uint32_t k, const ElementMatrix &dots) const { return volume * dots[j][k] * Phig[i]; }
    double D(uint32_t i, uint32_t j, uint32_t k, uint32_t l, const ElementMatrix &dots) const { return volume * dots[i][j] * dots[k][l]; }
};

// Create the St.Venant-Kirchhoff A,B,C,D coefficients for a tetrahedral element.
std::vector<ElementData> StVKABCD(const TetMesh *tets) {
    std::vector<ElementData> elements_data(tets->getNumElements());
    dvec3 columns[2];
    for (uint32_t el = 0; el < elements_data.size(); el++) {
        auto &element_data = elements_data[el];
        // Create the element data structure for a tet
        const double det = TetMesh::getTetDeterminant(tets->getVertex(el, 0), tets->getVertex(el, 1), tets->getVertex(el, 2), tets->getVertex(el, 3));
        element_data.volume = fabs(det / 6);
        for (uint32_t i = 0; i < NEV; i++) {
            for (uint32_t j = 0; j < 3; j++) {
                uint32_t ni = 0;
                for (uint32_t ii = 0; ii < NEV; ii++) {
                    if (ii == i) continue;

                    uint32_t nj = 0;
                    for (uint32_t jj = 0; jj < 3; jj++) {
                        if (jj != j) {
                            columns[nj][ni] = tets->getVertex(el, ii)[jj];
                            nj++;
                        }
                    }
                    const int sign = (i + j) % 2 == 0 ? -1 : 1;
                    element_data.Phig[i][j] = sign * dot(dvec3(1, 1, 1), cross(columns[0], columns[1])) / det;
                    ni++;
                }
            }
        }
    }
    return elements_data;
}
} // namespace

SparseMatrix StVKStiffnessMatrix::ComputeStiffnessMatrix(const TetMesh *tets, const double *vertex_displacements) {
    const uint32_t ne = tets->getNumElements();
    const double lambda = tets->material.getLambda();
    const double mu = tets->material.getMu();
    const auto outline = GetStiffnessMatrixTopology(*tets);
    SparseMatrix stiffness{outline};
    std::vector<int> vertices(NEV);
    auto **siffness_data = stiffness.GetDataHandle();
    // Build acceleration indices
    // auto **column_ = (int **)malloc(sizeof(int *) * ne);
    std::vector<std::array<int, NEV * NEV>> column_(ne);
    for (uint32_t el = 0; el < ne; el++) {
        // Seek for value row[j] in list associated with row[i]
        for (uint32_t i = 0; i < NEV; i++) {
            for (uint32_t j = 0; j < NEV; j++) {
                column_[el][NEV * i + j] = stiffness.GetInverseIndex(3 * tets->getVertexIndex(el, i), 3 * tets->getVertexIndex(el, j)) / 3;
            }
        }
    }
    dmat3 el_mat;
    auto precomputed_integrals = StVKABCD(tets);
    ElementMatrix dots; // cache dot products
    for (uint32_t el = 0; el < ne; el++) {
        const auto &ed = precomputed_integrals[el];
        for (uint32_t i = 0; i < NEV; i++) {
            for (uint32_t j = 0; j < NEV; j++) {
                dots[i][j] = dot(ed.Phig[i], ed.Phig[j]);
            }
        }

        for (uint32_t v = 0; v < NEV; v++) vertices[v] = tets->getVertexIndex(el, v);
        for (uint32_t c = 0; c < NEV; c++) {
            const uint32_t rowc = 3 * tets->getVertexIndex(el, c);
            for (uint32_t a = 0; a < NEV; a++) {
                const auto *qav = &(vertex_displacements[3 * vertices[a]]);
                const dvec3 qa{qav[0], qav[1], qav[2]};
                { // Linear terms
                    el_mat = {mu * ed.B(a, c, dots)}; // diag
                    el_mat += lambda * ed.A(c, a) + mu * ed.A(a, c);
                    // Add a 3x3 block matrix corresponding to a derivative of force on vertex c wrt to vertex a
                    for (uint32_t k = 0; k < 3; k++) {
                        for (uint32_t l = 0; l < 3; l++) {
                            stiffness.AddEntry(rowc + k, 3 * column_[el][NEV * c + a] + l, el_mat[k][l]);
                        }
                    }
                }
                { // Quadratic terms
                    el_mat = {0};
                    for (uint32_t e = 0; e < NEV; e++) {
                        const dvec3 C0 = lambda * ed.C(c, a, e, dots) + mu * (ed.C(e, a, c, dots) + ed.C(a, e, c, dots));
                        const dvec3 C1 = lambda * ed.C(e, a, c, dots) + mu * (ed.C(c, e, a, dots) + ed.C(a, e, c, dots));
                        const dvec3 C2 = lambda * ed.C(a, e, c, dots) + mu * (ed.C(c, a, e, dots) + ed.C(e, a, c, dots));
                        el_mat += glm::outerProduct(C0, qa) + glm::outerProduct(qa, C1) + dmat3{dot(qa, C2)};
                    }
                    // Add matrix block
                    for (uint32_t k = 0; k < 3; k++) {
                        for (uint32_t l = 0; l < 3; l++) {
                            siffness_data[rowc + k][3 * column_[el][NEV * c + a] + l] += el_mat[k][l];
                        }
                    }
                }
                { // Cubic terms
                  // Compute derivative on force on vertex c
                    el_mat = {0};
                    for (uint32_t e = 0; e < NEV; e++) {
                        for (uint32_t b = 0; b < NEV; b++) {
                            const double D0 = lambda * ed.D(a, c, b, e, dots) + mu * (ed.D(a, e, b, c, dots) + ed.D(a, b, c, e, dots));
                            const double D1 = 0.5 * lambda * ed.D(a, b, c, e, dots) + mu * ed.D(a, c, b, e, dots);
                            const auto *qbv = &(vertex_displacements[3 * vertices[b]]);
                            const dvec3 qb{qbv[0], qbv[1], qbv[2]};
                            el_mat += D0 * glm::outerProduct(qa, qb) + dmat3{dot(qa, qb) * D1};
                        }
                    }
                    // Add matrix block
                    for (uint32_t k = 0; k < 3; k++) {
                        for (uint32_t l = 0; l < 3; l++) {
                            siffness_data[rowc + k][3 * column_[el][NEV * c + a] + l] += el_mat[k][l];
                        }
                    }
                }
            }
        }
    }

    return stiffness;
}
