#pragma once

struct Vec3d {
    Vec3d() {}
    Vec3d(double x, double y, double z) {
        elt[0] = x;
        elt[1] = y;
        elt[2] = z;
    }
    Vec3d(const double *v) {
        elt[0] = v[0];
        elt[1] = v[1];
        elt[2] = v[2];
    }
    double &operator[](int index) { return elt[index]; }
    const double &operator[](int index) const { return elt[index]; }

    const Vec3d operator+(const Vec3d &vec2) const {
        auto sum = *this;
        sum.elt[0] += vec2.elt[0];
        sum.elt[1] += vec2.elt[1];
        sum.elt[2] += vec2.elt[2];
        return sum;
    }
    const Vec3d operator-(const Vec3d &vec2) const {
        auto sum = *this;
        sum.elt[0] -= vec2.elt[0];
        sum.elt[1] -= vec2.elt[1];
        sum.elt[2] -= vec2.elt[2];
        return sum;
    }
    Vec3d &operator+=(const Vec3d &vec2) {
        elt[0] += vec2.elt[0];
        elt[1] += vec2.elt[1];
        elt[2] += vec2.elt[2];
        return *this;
    }
    Vec3d &operator-=(const Vec3d &vec2) {
        elt[0] -= vec2.elt[0];
        elt[1] -= vec2.elt[1];
        elt[2] -= vec2.elt[2];
        return *this;
    }
    Vec3d &operator*=(double scalar) {
        elt[0] *= scalar;
        elt[1] *= scalar;
        elt[2] *= scalar;
        return *this;
    }

    friend Vec3d operator*(double scalar, const Vec3d &vec2) {
        auto result = vec2;
        result.elt[0] *= scalar;
        result.elt[1] *= scalar;
        result.elt[2] *= scalar;
        return result;
    }

    friend double dot(const Vec3d &vec1, const Vec3d &vec2) {
        return (vec1.elt[0] * vec2.elt[0] + vec1.elt[1] * vec2.elt[1] + vec1.elt[2] * vec2.elt[2]);
    }
    friend Vec3d cross(const Vec3d &vec1, const Vec3d &vec2) {
        return {vec1.elt[1] * vec2.elt[2] - vec2.elt[1] * vec1.elt[2], -vec1.elt[0] * vec2.elt[2] + vec2.elt[0] * vec1.elt[2], vec1.elt[0] * vec2.elt[1] - vec2.elt[0] * vec1.elt[1]};
    }

private:
    double elt[3];
};
