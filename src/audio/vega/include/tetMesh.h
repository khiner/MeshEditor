#pragma once

// Container for a tetrahedral volumetric 3D mesh.

#include "vec3d.h"

#include <memory.h>
#include <stdlib.h>

struct TetMesh {
    TetMesh(int numVertices_, double *vertices_, int numElements_, int *elements_, double E, double nu, double density)
        : material(density, E, nu), numElementVertices(4) {
        numElements = numElements_;
        numVertices = numVertices_;

        vertices = new Vec3d[numVertices];
        elements = (int **)malloc(sizeof(int *) * numElements);

        for (int i = 0; i < numVertices; i++) vertices[i] = {vertices_[3 * i + 0], vertices_[3 * i + 1], vertices_[3 * i + 2]};

        int *v = (int *)malloc(sizeof(int) * numElementVertices);
        for (int i = 0; i < numElements; i++) {
            elements[i] = (int *)malloc(sizeof(int) * numElementVertices);
            for (int j = 0; j < numElementVertices; j++) {
                v[j] = elements_[numElementVertices * i + j];
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

    static double getTetVolume(const Vec3d &a, const Vec3d &b, const Vec3d &c, const Vec3d &d);
    static double getTetDeterminant(const Vec3d &a, const Vec3d &b, const Vec3d &c, const Vec3d &d);

    int getNumVertices() const { return numVertices; }
    Vec3d &getVertex(int i) { return vertices[i]; }
    const Vec3d &getVertex(int i) const { return vertices[i]; }
    Vec3d &getVertex(int element, int vertex) { return vertices[elements[element][vertex]]; }
    const Vec3d &getVertex(int element, int vertex) const { return vertices[elements[element][vertex]]; }
    int getVertexIndex(int element, int vertex) const { return elements[element][vertex]; }
    int getNumElements() const { return numElements; }
    int getNumElementVertices() const { return numElementVertices; }
    void setVertex(int i, const Vec3d &pos) { vertices[i] = pos; } // set the position of a vertex

    // mass density of an element
    double getElementDensity(int el) const { return material.getDensity(); }
    // computes the mass matrix of a single element
    // note: to compute the mass matrix for the entire mesh, use generateMassMatrix.h
    void computeElementMassMatrix(int element, double *massMatrix) const; // massMatrix is numElementVertices_ x numElementVertices_
    // center of mass and inertia tensor
    double getElementVolume(int el) const;

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

        double invNumElementVertices = 1.0 / getNumElementVertices();
        for (int el = 0; el < numElements; el++) {
            double volume = getElementVolume(el);
            double density = getElementDensity(el);
            double mass = density * volume;
            for (int j = 0; j < getNumElementVertices(); j++)
                gravityForce[3 * getVertexIndex(el, j) + 1] -= invNumElementVertices * mass * g; // gravity assumed to act in negative y-direction
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
    Vec3d *vertices;

    int numElementVertices;
    int numElements;
    int **elements;
};
