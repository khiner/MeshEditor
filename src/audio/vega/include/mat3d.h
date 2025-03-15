#pragma once

#include "vec3d.h"

struct Mat3d {
    Mat3d() {}

    /*
            [ x0  x1  x2 ]
      M  =  [ x3  x4  x5 ]
            [ x6  x7  x8 ]
    */
    Mat3d(double x0_g, double x1_g, double x2_g, double x3_g, double x4_g, double x5_g, double x6_g, double x7_g, double x8_g) {
        elt[0] = {x0_g, x1_g, x2_g};
        elt[1] = {x3_g, x4_g, x5_g};
        elt[2] = {x6_g, x7_g, x8_g};
    }
    Mat3d(double diag) {
        elt[0] = {diag, 0, 0};
        elt[1] = {0, diag, 0};
        elt[2] = {0, 0, diag};
    }

    const Vec3d &operator[](int index) const { return elt[index]; }
    Vec3d &operator[](int index) { return elt[index]; }

    Mat3d operator+(const Mat3d &mat2) const {
        auto sum = *this;
        sum.elt[0] += mat2.elt[0];
        sum.elt[1] += mat2.elt[1];
        sum.elt[2] += mat2.elt[2];
        return sum;
    }
    Mat3d &operator+=(const Mat3d &mat2) {
        elt[0] += mat2.elt[0];
        elt[1] += mat2.elt[1];
        elt[2] += mat2.elt[2];
        return *this;
    }
    Mat3d operator-(const Mat3d &mat2) const {
        auto sum = *this;
        sum.elt[0] -= mat2.elt[0];
        sum.elt[1] -= mat2.elt[1];
        sum.elt[2] -= mat2.elt[2];
        return sum;
    }
    Mat3d &operator*=(double scalar) {
        elt[0] *= scalar;
        elt[1] *= scalar;
        elt[2] *= scalar;
        return *this;
    }

    friend Mat3d operator*(double scalar, const Mat3d &mat2) {
        Mat3d result = mat2;
        result.elt[0] *= scalar;
        result.elt[1] *= scalar;
        result.elt[2] *= scalar;
        return result;
    }

    const Vec3d operator*(const Vec3d &vec) const {
        return {dot(elt[0], vec), dot(elt[1], vec), dot(elt[2], vec)};
    }
    const Mat3d operator*(const Mat3d &mat2) const {
        return {
            dot(elt[0], {mat2.elt[0][0], mat2.elt[1][0], mat2.elt[2][0]}),
            dot(elt[0], {mat2.elt[0][1], mat2.elt[1][1], mat2.elt[2][1]}),
            dot(elt[0], {mat2.elt[0][2], mat2.elt[1][2], mat2.elt[2][2]}),

            dot(elt[1], {mat2.elt[0][0], mat2.elt[1][0], mat2.elt[2][0]}),
            dot(elt[1], {mat2.elt[0][1], mat2.elt[1][1], mat2.elt[2][1]}),
            dot(elt[1], {mat2.elt[0][2], mat2.elt[1][2], mat2.elt[2][2]}),

            dot(elt[2], {mat2.elt[0][0], mat2.elt[1][0], mat2.elt[2][0]}),
            dot(elt[2], {mat2.elt[0][1], mat2.elt[1][1], mat2.elt[2][1]}),
            dot(elt[2], {mat2.elt[0][2], mat2.elt[1][2], mat2.elt[2][2]})
        };
    }

private:
    Vec3d elt[3]; // the three rows of the matrix
};

inline static Mat3d tensorProduct(const Vec3d &vecA, const Vec3d &vecB) {
    return {vecA[0] * vecB[0], vecA[0] * vecB[1], vecA[0] * vecB[2], vecA[1] * vecB[0], vecA[1] * vecB[1], vecA[1] * vecB[2], vecA[2] * vecB[0], vecA[2] * vecB[1], vecA[2] * vecB[2]};
}
