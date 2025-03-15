#pragma once

// Container for a tetrahedral volumetric 3D mesh.

#include "numeric/vec3.h"

#include <glm/geometric.hpp>

#include <cmath>
#include <memory.h>
#include <stdlib.h>

struct TetMesh {
    static constexpr uint32_t NumElementVertices = 4;

    TetMesh(int numVertices_, double *vertices_, int numElements_, int *elements_, double E, double nu, double density)
        : material(density, E, nu), numElements(numElements_), numVertices(numVertices_),
          vertices(new dvec3[numVertices]), elements((int **)malloc(sizeof(int *) * numElements)) {
        for (int i = 0; i < numVertices; i++) vertices[i] = {vertices_[3 * i + 0], vertices_[3 * i + 1], vertices_[3 * i + 2]};

        int *v = (int *)malloc(sizeof(int) * NumElementVertices);
        for (int i = 0; i < numElements; i++) {
            elements[i] = (int *)malloc(sizeof(int) * NumElementVertices);
            for (int j = 0; j < NumElementVertices; j++) {
                v[j] = elements_[NumElementVertices * i + j];
                elements[i][j] = v[j];
            }
        }
        free(v);
    }

    ~TetMesh() {
        delete[] vertices;
        for (int i = 0; i < numElements; i++) free(elements[i]);
        free(elements);
    }

    // volume = 1/6 * |(d-a) . ((b-a) x (c-a))|
    inline static double getTetVolume(const dvec3 &a, const dvec3 &b, const dvec3 &c, const dvec3 &d) {
        return (1.f / 6.f) * fabs(getTetDeterminant(a, b, c, d));
    }
    inline static double getTetDeterminant(const dvec3 &a, const dvec3 &b, const dvec3 &c, const dvec3 &d) {
        // When det(A) > 0, tet has positive orientation.
        // When det(A) = 0, tet is degenerate.
        // When det(A) < 0, tet has negative orientation.
        return glm::dot(d - a, glm::cross(b - a, c - a));
    }

    int getNumVertices() const { return numVertices; }
    dvec3 &getVertex(int i) { return vertices[i]; }
    const dvec3 &getVertex(int i) const { return vertices[i]; }
    dvec3 &getVertex(int element, int vertex) { return vertices[elements[element][vertex]]; }
    const dvec3 &getVertex(int element, int vertex) const { return vertices[elements[element][vertex]]; }
    int getVertexIndex(int element, int vertex) const { return elements[element][vertex]; }
    int getNumElements() const { return numElements; }
    void setVertex(int i, const dvec3 &pos) { vertices[i] = pos; } // set the position of a vertex

    // mass density of an element
    double getElementDensity(int el) const { return material.getDensity(); }
    double getElementVolume(int el) const {
        return getTetVolume(getVertex(el, 0), getVertex(el, 1), getVertex(el, 2), getVertex(el, 3));
    }

    double getMass() const {
        double mass = 0.0;
        double density = material.getDensity();
        for (int i = 0; i < numElements; i++) mass += getElementVolume(i) * density;
        return mass;
    }

    // computes the gravity vector (different forces on different mesh vertices due to potentially varying mass densities)
    // gravityForce must be a pre-allocated vector of length 3xnumVertices()
    void computeGravity(double *gravityForce, double g = 9.81, bool addForce = false) const {
        if (!addForce) memset(gravityForce, 0, sizeof(double) * 3 * numVertices);
        static constexpr double InvNumElementVertices = 1.0 / NumElementVertices;
        for (int el = 0; el < numElements; el++) {
            const double mass = getElementDensity(el) * getElementVolume(el);
            for (int j = 0; j < NumElementVertices; j++) {
                gravityForce[3 * getVertexIndex(el, j) + 1] -= InvNumElementVertices * mass * g; // gravity assumed to act in negative y-direction
            }
        }
    }

    // stores an isotropic material specified by E (Young's modulus), nu (Poisson's ratio), and density
    // such a material specification is very common: (corotational) linear FEM, StVK, etc.
    struct ENuMaterial {
        ENuMaterial(double density, double E, double nu) : density(density), E_(E), nu_(nu) {}
        ~ENuMaterial() {}

        double getDensity() const { return density; }
        double getE() const { return E_; }
        double getNu() const { return nu_; }
        // Lame's lambda coefficient
        double getLambda() const { return (nu_ * E_) / ((1 + nu_) * (1 - 2 * nu_)); }
        // Lame's mu coefficient
        double getMu() const { return E_ / (2 * (1 + nu_)); }
        void setE(double E) { E_ = E; }
        void setNu(double nu) { nu_ = nu; }

    private:
        double density, E_, nu_; // density, Young's modulus, Poisson's ratio
    };

    ENuMaterial material;

private:
    int numVertices;
    dvec3 *vertices;

    int numElements;
    int **elements;
};
