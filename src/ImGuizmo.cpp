#ifndef IMGUI_DEFINE_MATH_OPERATORS
#define IMGUI_DEFINE_MATH_OPERATORS
#endif

#include "ImGuizmo.h"
#include "imgui.h"
#include "imgui_internal.h"

#include <string_view>

namespace ImGuizmo {
struct Style {
    Style();

    float TranslationLineThickness; // Thickness of lines for translation gizmo
    float TranslationLineArrowSize; // Size of arrow at the end of lines for translation gizmo
    float RotationLineThickness; // Thickness of lines for rotation gizmo
    float RotationOuterLineThickness; // Thickness of line surrounding the rotation gizmo
    float ScaleLineThickness; // Thickness of lines for scale gizmo
    float ScaleLineCircleSize; // Size of circle at the end of lines for scale gizmo
    float HatchedAxisLineThickness; // Thickness of hatched axis lines
    float CenterCircleSize; // Size of circle at the center of the translate/scale gizmo

    ImVec4 Colors[COLOR::COUNT];
};

static const float ZPI = 3.14159265358979323846f;
static const float DEG2RAD = (ZPI / 180.f);
const float screenRotateSize = 0.06f;
// scale a bit so translate axis do not touch when in universal
const float rotationDisplayFactor = 1.2f;

static bool Intersects(OPERATION lhs, OPERATION rhs) {
    return (lhs & rhs) != 0;
}

// True if lhs contains rhs
static bool Contains(OPERATION lhs, OPERATION rhs) {
    return (lhs & rhs) == rhs;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// utility and math

void FPU_MatrixF_x_MatrixF(const float *a, const float *b, float *r) {
    r[0] = a[0] * b[0] + a[1] * b[4] + a[2] * b[8] + a[3] * b[12];
    r[1] = a[0] * b[1] + a[1] * b[5] + a[2] * b[9] + a[3] * b[13];
    r[2] = a[0] * b[2] + a[1] * b[6] + a[2] * b[10] + a[3] * b[14];
    r[3] = a[0] * b[3] + a[1] * b[7] + a[2] * b[11] + a[3] * b[15];

    r[4] = a[4] * b[0] + a[5] * b[4] + a[6] * b[8] + a[7] * b[12];
    r[5] = a[4] * b[1] + a[5] * b[5] + a[6] * b[9] + a[7] * b[13];
    r[6] = a[4] * b[2] + a[5] * b[6] + a[6] * b[10] + a[7] * b[14];
    r[7] = a[4] * b[3] + a[5] * b[7] + a[6] * b[11] + a[7] * b[15];

    r[8] = a[8] * b[0] + a[9] * b[4] + a[10] * b[8] + a[11] * b[12];
    r[9] = a[8] * b[1] + a[9] * b[5] + a[10] * b[9] + a[11] * b[13];
    r[10] = a[8] * b[2] + a[9] * b[6] + a[10] * b[10] + a[11] * b[14];
    r[11] = a[8] * b[3] + a[9] * b[7] + a[10] * b[11] + a[11] * b[15];

    r[12] = a[12] * b[0] + a[13] * b[4] + a[14] * b[8] + a[15] * b[12];
    r[13] = a[12] * b[1] + a[13] * b[5] + a[14] * b[9] + a[15] * b[13];
    r[14] = a[12] * b[2] + a[13] * b[6] + a[14] * b[10] + a[15] * b[14];
    r[15] = a[12] * b[3] + a[13] * b[7] + a[14] * b[11] + a[15] * b[15];
}

template<typename T> T Clamp(T x, T y, T z) { return ((x < y) ? y : ((x > z) ? z : x)); }
template<typename T> T max(T x, T y) { return (x > y) ? x : y; }
template<typename T> T min(T x, T y) { return (x < y) ? x : y; }
template<typename T> bool IsWithin(T x, T y, T z) { return (x >= y) && (x <= z); }

struct matrix_t;
struct vec_t {
public:
    float x, y, z, w;

    void Set(float v) { x = y = z = w = v; }
    void Set(float _x, float _y, float _z = 0.f, float _w = 0.f) {
        x = _x;
        y = _y;
        z = _z;
        w = _w;
    }

    vec_t &operator-=(const vec_t &v) {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        w -= v.w;
        return *this;
    }
    vec_t &operator+=(const vec_t &v) {
        x += v.x;
        y += v.y;
        z += v.z;
        w += v.w;
        return *this;
    }
    vec_t &operator*=(const vec_t &v) {
        x *= v.x;
        y *= v.y;
        z *= v.z;
        w *= v.w;
        return *this;
    }
    vec_t &operator*=(float v) {
        x *= v;
        y *= v;
        z *= v;
        w *= v;
        return *this;
    }

    vec_t operator*(float f) const;
    vec_t operator-() const;
    vec_t operator-(const vec_t &v) const;
    vec_t operator+(const vec_t &v) const;
    vec_t operator*(const vec_t &v) const;

    const vec_t &operator+() const { return (*this); }
    float Length() const { return sqrtf(x * x + y * y + z * z); };
    float LengthSq() const { return (x * x + y * y + z * z); };
    vec_t Normalize() {
        (*this) *= (1.f / (Length() > FLT_EPSILON ? Length() : FLT_EPSILON));
        return (*this);
    }
    vec_t Normalize(const vec_t &v) {
        this->Set(v.x, v.y, v.z, v.w);
        this->Normalize();
        return (*this);
    }

    void Cross(const vec_t &v) {
        vec_t res;
        res.x = y * v.z - z * v.y;
        res.y = z * v.x - x * v.z;
        res.z = x * v.y - y * v.x;

        x = res.x;
        y = res.y;
        z = res.z;
        w = 0.f;
    }

    void Cross(const vec_t &v1, const vec_t &v2) {
        x = v1.y * v2.z - v1.z * v2.y;
        y = v1.z * v2.x - v1.x * v2.z;
        z = v1.x * v2.y - v1.y * v2.x;
        w = 0.f;
    }

    float Dot(const vec_t &v) const {
        return (x * v.x) + (y * v.y) + (z * v.z) + (w * v.w);
    }

    float Dot3(const vec_t &v) const {
        return (x * v.x) + (y * v.y) + (z * v.z);
    }

    void Transform(const matrix_t &matrix);
    void Transform(const vec_t &s, const matrix_t &matrix);

    void TransformVector(const matrix_t &matrix);
    void TransformPoint(const matrix_t &matrix);
    void TransformVector(const vec_t &v, const matrix_t &matrix) {
        (*this) = v;
        this->TransformVector(matrix);
    }
    void TransformPoint(const vec_t &v, const matrix_t &matrix) {
        (*this) = v;
        this->TransformPoint(matrix);
    }

    float &operator[](size_t index) { return ((float *)&x)[index]; }
    const float &operator[](size_t index) const { return ((float *)&x)[index]; }
    bool operator!=(const vec_t &other) const { return memcmp(this, &other, sizeof(vec_t)) != 0; }
};

vec_t makeVect(float _x, float _y, float _z = 0.f, float _w = 0.f) {
    vec_t res;
    res.x = _x;
    res.y = _y;
    res.z = _z;
    res.w = _w;
    return res;
}
vec_t makeVect(ImVec2 v) {
    vec_t res;
    res.x = v.x;
    res.y = v.y;
    res.z = 0.f;
    res.w = 0.f;
    return res;
}
vec_t vec_t::operator*(float f) const { return makeVect(x * f, y * f, z * f, w * f); }
vec_t vec_t::operator-() const { return makeVect(-x, -y, -z, -w); }
vec_t vec_t::operator-(const vec_t &v) const { return makeVect(x - v.x, y - v.y, z - v.z, w - v.w); }
vec_t vec_t::operator+(const vec_t &v) const { return makeVect(x + v.x, y + v.y, z + v.z, w + v.w); }
vec_t vec_t::operator*(const vec_t &v) const { return makeVect(x * v.x, y * v.y, z * v.z, w * v.w); }

vec_t Normalized(const vec_t &v) {
    vec_t res;
    res = v;
    res.Normalize();
    return res;
}
vec_t Cross(const vec_t &v1, const vec_t &v2) {
    vec_t res;
    res.x = v1.y * v2.z - v1.z * v2.y;
    res.y = v1.z * v2.x - v1.x * v2.z;
    res.z = v1.x * v2.y - v1.y * v2.x;
    res.w = 0.f;
    return res;
}

float Dot(const vec_t &v1, const vec_t &v2) {
    return (v1.x * v2.x) + (v1.y * v2.y) + (v1.z * v2.z);
}

vec_t BuildPlan(const vec_t &p_point1, const vec_t &p_normal) {
    vec_t normal, res;
    normal.Normalize(p_normal);
    res.w = normal.Dot(p_point1);
    res.x = normal.x;
    res.y = normal.y;
    res.z = normal.z;
    return res;
}

struct matrix_t {
public:
    union {
        float m[4][4];
        float m16[16];
        struct
        {
            vec_t right, up, dir, position;
        } v;
        vec_t component[4];
    };

    operator float *() { return m16; }
    operator const float *() const { return m16; }
    void Translation(float _x, float _y, float _z) { this->Translation(makeVect(_x, _y, _z)); }

    void Translation(const vec_t &vt) {
        v.right.Set(1.f, 0.f, 0.f, 0.f);
        v.up.Set(0.f, 1.f, 0.f, 0.f);
        v.dir.Set(0.f, 0.f, 1.f, 0.f);
        v.position.Set(vt.x, vt.y, vt.z, 1.f);
    }

    void Scale(float _x, float _y, float _z) {
        v.right.Set(_x, 0.f, 0.f, 0.f);
        v.up.Set(0.f, _y, 0.f, 0.f);
        v.dir.Set(0.f, 0.f, _z, 0.f);
        v.position.Set(0.f, 0.f, 0.f, 1.f);
    }
    void Scale(const vec_t &s) { Scale(s.x, s.y, s.z); }

    matrix_t &operator*=(const matrix_t &mat) {
        matrix_t tmpMat;
        tmpMat = *this;
        tmpMat.Multiply(mat);
        *this = tmpMat;
        return *this;
    }
    matrix_t operator*(const matrix_t &mat) const {
        matrix_t matT;
        matT.Multiply(*this, mat);
        return matT;
    }

    void Multiply(const matrix_t &matrix) {
        matrix_t tmp;
        tmp = *this;

        FPU_MatrixF_x_MatrixF((float *)&tmp, (float *)&matrix, (float *)this);
    }

    void Multiply(const matrix_t &m1, const matrix_t &m2) {
        FPU_MatrixF_x_MatrixF((float *)&m1, (float *)&m2, (float *)this);
    }

    float GetDeterminant() const {
        return m[0][0] * m[1][1] * m[2][2] + m[0][1] * m[1][2] * m[2][0] + m[0][2] * m[1][0] * m[2][1] -
            m[0][2] * m[1][1] * m[2][0] - m[0][1] * m[1][0] * m[2][2] - m[0][0] * m[1][2] * m[2][1];
    }

    float Inverse(const matrix_t &srcMatrix, bool affine = false);
    void SetToIdentity() {
        v.right.Set(1.f, 0.f, 0.f, 0.f);
        v.up.Set(0.f, 1.f, 0.f, 0.f);
        v.dir.Set(0.f, 0.f, 1.f, 0.f);
        v.position.Set(0.f, 0.f, 0.f, 1.f);
    }
    void Transpose() {
        matrix_t tmpm;
        for (int l = 0; l < 4; l++) {
            for (int c = 0; c < 4; c++) {
                tmpm.m[l][c] = m[c][l];
            }
        }
        (*this) = tmpm;
    }

    void RotationAxis(const vec_t &axis, float angle);

    void OrthoNormalize() {
        v.right.Normalize();
        v.up.Normalize();
        v.dir.Normalize();
    }
};

void vec_t::Transform(const matrix_t &matrix) {
    vec_t out;

    out.x = x * matrix.m[0][0] + y * matrix.m[1][0] + z * matrix.m[2][0] + w * matrix.m[3][0];
    out.y = x * matrix.m[0][1] + y * matrix.m[1][1] + z * matrix.m[2][1] + w * matrix.m[3][1];
    out.z = x * matrix.m[0][2] + y * matrix.m[1][2] + z * matrix.m[2][2] + w * matrix.m[3][2];
    out.w = x * matrix.m[0][3] + y * matrix.m[1][3] + z * matrix.m[2][3] + w * matrix.m[3][3];

    x = out.x;
    y = out.y;
    z = out.z;
    w = out.w;
}

void vec_t::Transform(const vec_t &s, const matrix_t &matrix) {
    *this = s;
    Transform(matrix);
}

void vec_t::TransformPoint(const matrix_t &matrix) {
    vec_t out;

    out.x = x * matrix.m[0][0] + y * matrix.m[1][0] + z * matrix.m[2][0] + matrix.m[3][0];
    out.y = x * matrix.m[0][1] + y * matrix.m[1][1] + z * matrix.m[2][1] + matrix.m[3][1];
    out.z = x * matrix.m[0][2] + y * matrix.m[1][2] + z * matrix.m[2][2] + matrix.m[3][2];
    out.w = x * matrix.m[0][3] + y * matrix.m[1][3] + z * matrix.m[2][3] + matrix.m[3][3];

    x = out.x;
    y = out.y;
    z = out.z;
    w = out.w;
}

void vec_t::TransformVector(const matrix_t &matrix) {
    vec_t out;

    out.x = x * matrix.m[0][0] + y * matrix.m[1][0] + z * matrix.m[2][0];
    out.y = x * matrix.m[0][1] + y * matrix.m[1][1] + z * matrix.m[2][1];
    out.z = x * matrix.m[0][2] + y * matrix.m[1][2] + z * matrix.m[2][2];
    out.w = x * matrix.m[0][3] + y * matrix.m[1][3] + z * matrix.m[2][3];

    x = out.x;
    y = out.y;
    z = out.z;
    w = out.w;
}

float matrix_t::Inverse(const matrix_t &srcMatrix, bool affine) {
    float det = 0;

    if (affine) {
        det = GetDeterminant();
        float s = 1 / det;
        m[0][0] = (srcMatrix.m[1][1] * srcMatrix.m[2][2] - srcMatrix.m[1][2] * srcMatrix.m[2][1]) * s;
        m[0][1] = (srcMatrix.m[2][1] * srcMatrix.m[0][2] - srcMatrix.m[2][2] * srcMatrix.m[0][1]) * s;
        m[0][2] = (srcMatrix.m[0][1] * srcMatrix.m[1][2] - srcMatrix.m[0][2] * srcMatrix.m[1][1]) * s;
        m[1][0] = (srcMatrix.m[1][2] * srcMatrix.m[2][0] - srcMatrix.m[1][0] * srcMatrix.m[2][2]) * s;
        m[1][1] = (srcMatrix.m[2][2] * srcMatrix.m[0][0] - srcMatrix.m[2][0] * srcMatrix.m[0][2]) * s;
        m[1][2] = (srcMatrix.m[0][2] * srcMatrix.m[1][0] - srcMatrix.m[0][0] * srcMatrix.m[1][2]) * s;
        m[2][0] = (srcMatrix.m[1][0] * srcMatrix.m[2][1] - srcMatrix.m[1][1] * srcMatrix.m[2][0]) * s;
        m[2][1] = (srcMatrix.m[2][0] * srcMatrix.m[0][1] - srcMatrix.m[2][1] * srcMatrix.m[0][0]) * s;
        m[2][2] = (srcMatrix.m[0][0] * srcMatrix.m[1][1] - srcMatrix.m[0][1] * srcMatrix.m[1][0]) * s;
        m[3][0] = -(m[0][0] * srcMatrix.m[3][0] + m[1][0] * srcMatrix.m[3][1] + m[2][0] * srcMatrix.m[3][2]);
        m[3][1] = -(m[0][1] * srcMatrix.m[3][0] + m[1][1] * srcMatrix.m[3][1] + m[2][1] * srcMatrix.m[3][2]);
        m[3][2] = -(m[0][2] * srcMatrix.m[3][0] + m[1][2] * srcMatrix.m[3][1] + m[2][2] * srcMatrix.m[3][2]);
    } else {
        // transpose matrix
        float src[16];
        for (int i = 0; i < 4; ++i) {
            src[i] = srcMatrix.m16[i * 4];
            src[i + 4] = srcMatrix.m16[i * 4 + 1];
            src[i + 8] = srcMatrix.m16[i * 4 + 2];
            src[i + 12] = srcMatrix.m16[i * 4 + 3];
        }

        // calculate pairs for first 8 elements (cofactors)
        float tmp[12]; // temp array for pairs
        tmp[0] = src[10] * src[15];
        tmp[1] = src[11] * src[14];
        tmp[2] = src[9] * src[15];
        tmp[3] = src[11] * src[13];
        tmp[4] = src[9] * src[14];
        tmp[5] = src[10] * src[13];
        tmp[6] = src[8] * src[15];
        tmp[7] = src[11] * src[12];
        tmp[8] = src[8] * src[14];
        tmp[9] = src[10] * src[12];
        tmp[10] = src[8] * src[13];
        tmp[11] = src[9] * src[12];

        // calculate first 8 elements (cofactors)
        m16[0] = (tmp[0] * src[5] + tmp[3] * src[6] + tmp[4] * src[7]) - (tmp[1] * src[5] + tmp[2] * src[6] + tmp[5] * src[7]);
        m16[1] = (tmp[1] * src[4] + tmp[6] * src[6] + tmp[9] * src[7]) - (tmp[0] * src[4] + tmp[7] * src[6] + tmp[8] * src[7]);
        m16[2] = (tmp[2] * src[4] + tmp[7] * src[5] + tmp[10] * src[7]) - (tmp[3] * src[4] + tmp[6] * src[5] + tmp[11] * src[7]);
        m16[3] = (tmp[5] * src[4] + tmp[8] * src[5] + tmp[11] * src[6]) - (tmp[4] * src[4] + tmp[9] * src[5] + tmp[10] * src[6]);
        m16[4] = (tmp[1] * src[1] + tmp[2] * src[2] + tmp[5] * src[3]) - (tmp[0] * src[1] + tmp[3] * src[2] + tmp[4] * src[3]);
        m16[5] = (tmp[0] * src[0] + tmp[7] * src[2] + tmp[8] * src[3]) - (tmp[1] * src[0] + tmp[6] * src[2] + tmp[9] * src[3]);
        m16[6] = (tmp[3] * src[0] + tmp[6] * src[1] + tmp[11] * src[3]) - (tmp[2] * src[0] + tmp[7] * src[1] + tmp[10] * src[3]);
        m16[7] = (tmp[4] * src[0] + tmp[9] * src[1] + tmp[10] * src[2]) - (tmp[5] * src[0] + tmp[8] * src[1] + tmp[11] * src[2]);

        // calculate pairs for second 8 elements (cofactors)
        tmp[0] = src[2] * src[7];
        tmp[1] = src[3] * src[6];
        tmp[2] = src[1] * src[7];
        tmp[3] = src[3] * src[5];
        tmp[4] = src[1] * src[6];
        tmp[5] = src[2] * src[5];
        tmp[6] = src[0] * src[7];
        tmp[7] = src[3] * src[4];
        tmp[8] = src[0] * src[6];
        tmp[9] = src[2] * src[4];
        tmp[10] = src[0] * src[5];
        tmp[11] = src[1] * src[4];

        // calculate second 8 elements (cofactors)
        m16[8] = (tmp[0] * src[13] + tmp[3] * src[14] + tmp[4] * src[15]) - (tmp[1] * src[13] + tmp[2] * src[14] + tmp[5] * src[15]);
        m16[9] = (tmp[1] * src[12] + tmp[6] * src[14] + tmp[9] * src[15]) - (tmp[0] * src[12] + tmp[7] * src[14] + tmp[8] * src[15]);
        m16[10] = (tmp[2] * src[12] + tmp[7] * src[13] + tmp[10] * src[15]) - (tmp[3] * src[12] + tmp[6] * src[13] + tmp[11] * src[15]);
        m16[11] = (tmp[5] * src[12] + tmp[8] * src[13] + tmp[11] * src[14]) - (tmp[4] * src[12] + tmp[9] * src[13] + tmp[10] * src[14]);
        m16[12] = (tmp[2] * src[10] + tmp[5] * src[11] + tmp[1] * src[9]) - (tmp[4] * src[11] + tmp[0] * src[9] + tmp[3] * src[10]);
        m16[13] = (tmp[8] * src[11] + tmp[0] * src[8] + tmp[7] * src[10]) - (tmp[6] * src[10] + tmp[9] * src[11] + tmp[1] * src[8]);
        m16[14] = (tmp[6] * src[9] + tmp[11] * src[11] + tmp[3] * src[8]) - (tmp[10] * src[11] + tmp[2] * src[8] + tmp[7] * src[9]);
        m16[15] = (tmp[10] * src[10] + tmp[4] * src[8] + tmp[9] * src[9]) - (tmp[8] * src[9] + tmp[11] * src[10] + tmp[5] * src[8]);

        // calculate determinant
        det = src[0] * m16[0] + src[1] * m16[1] + src[2] * m16[2] + src[3] * m16[3];

        // calculate matrix inverse
        float invdet = 1 / det;
        for (int j = 0; j < 16; ++j) {
            m16[j] *= invdet;
        }
    }

    return det;
}

void matrix_t::RotationAxis(const vec_t &axis, float angle) {
    float length2 = axis.LengthSq();
    if (length2 < FLT_EPSILON) {
        SetToIdentity();
        return;
    }

    vec_t n = axis * (1.f / sqrtf(length2));
    float s = sinf(angle);
    float c = cosf(angle);
    float k = 1.f - c;

    float xx = n.x * n.x * k + c;
    float yy = n.y * n.y * k + c;
    float zz = n.z * n.z * k + c;
    float xy = n.x * n.y * k;
    float yz = n.y * n.z * k;
    float zx = n.z * n.x * k;
    float xs = n.x * s;
    float ys = n.y * s;
    float zs = n.z * s;

    m[0][0] = xx;
    m[0][1] = xy + zs;
    m[0][2] = zx - ys;
    m[0][3] = 0.f;
    m[1][0] = xy - zs;
    m[1][1] = yy;
    m[1][2] = yz + xs;
    m[1][3] = 0.f;
    m[2][0] = zx + ys;
    m[2][1] = yz - xs;
    m[2][2] = zz;
    m[2][3] = 0.f;
    m[3][0] = 0.f;
    m[3][1] = 0.f;
    m[3][2] = 0.f;
    m[3][3] = 1.f;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//

enum MOVETYPE {
    MT_NONE,
    MT_MOVE_X,
    MT_MOVE_Y,
    MT_MOVE_Z,
    MT_MOVE_YZ,
    MT_MOVE_ZX,
    MT_MOVE_XY,
    MT_MOVE_SCREEN,
    MT_ROTATE_X,
    MT_ROTATE_Y,
    MT_ROTATE_Z,
    MT_ROTATE_SCREEN,
    MT_SCALE_X,
    MT_SCALE_Y,
    MT_SCALE_Z,
    MT_SCALE_XYZ
};

static bool IsTranslateType(int type) {
    return type >= MT_MOVE_X && type <= MT_MOVE_SCREEN;
}

static bool IsRotateType(int type) {
    return type >= MT_ROTATE_X && type <= MT_ROTATE_SCREEN;
}

static bool IsScaleType(int type) {
    return type >= MT_SCALE_X && type <= MT_SCALE_XYZ;
}

// Matches MT_MOVE_AB order
static const OPERATION TRANSLATE_PLANS[3] = {TRANSLATE_Y | TRANSLATE_Z, TRANSLATE_X | TRANSLATE_Z, TRANSLATE_X | TRANSLATE_Y};

Style::Style() {
    // default values
    TranslationLineThickness = 3.0f;
    TranslationLineArrowSize = 6.0f;
    RotationLineThickness = 2.0f;
    RotationOuterLineThickness = 3.0f;
    ScaleLineThickness = 3.0f;
    ScaleLineCircleSize = 6.0f;
    HatchedAxisLineThickness = 6.0f;
    CenterCircleSize = 6.0f;

    // initialize default colors
    Colors[DIRECTION_X] = ImVec4(0.666f, 0.000f, 0.000f, 1.000f);
    Colors[DIRECTION_Y] = ImVec4(0.000f, 0.666f, 0.000f, 1.000f);
    Colors[DIRECTION_Z] = ImVec4(0.000f, 0.000f, 0.666f, 1.000f);
    Colors[PLANE_X] = ImVec4(0.666f, 0.000f, 0.000f, 0.380f);
    Colors[PLANE_Y] = ImVec4(0.000f, 0.666f, 0.000f, 0.380f);
    Colors[PLANE_Z] = ImVec4(0.000f, 0.000f, 0.666f, 0.380f);
    Colors[SELECTION] = ImVec4(1.000f, 0.500f, 0.062f, 0.541f);
    Colors[INACTIVE] = ImVec4(0.600f, 0.600f, 0.600f, 0.600f);
    Colors[TRANSLATION_LINE] = ImVec4(0.666f, 0.666f, 0.666f, 0.666f);
    Colors[SCALE_LINE] = ImVec4(0.250f, 0.250f, 0.250f, 1.000f);
    Colors[ROTATION_USING_BORDER] = ImVec4(1.000f, 0.500f, 0.062f, 1.000f);
    Colors[ROTATION_USING_FILL] = ImVec4(1.000f, 0.500f, 0.062f, 0.500f);
    Colors[HATCHED_AXIS_LINES] = ImVec4(0.000f, 0.000f, 0.000f, 0.500f);
    Colors[TEXT] = ImVec4(1.000f, 1.000f, 1.000f, 1.000f);
    Colors[TEXT_SHADOW] = ImVec4(0.000f, 0.000f, 0.000f, 1.000f);
}

struct Context {
    Style mStyle;

    MODE mMode;
    matrix_t mViewMat;
    matrix_t mProjectionMat;
    matrix_t mModel;
    matrix_t mModelLocal; // orthonormalized model
    matrix_t mModelInverse;
    matrix_t mModelSource;
    matrix_t mMVP;
    matrix_t mMVPLocal; // MVP with full model matrix whereas mMVP's model matrix might only be translation in case of World space edition
    matrix_t mViewProjection;

    vec_t mModelScaleOrigin;
    vec_t mCameraEye;
    vec_t mCameraDir;
    vec_t mRayOrigin;
    vec_t mRayVector;

    float mRadiusSquareCenter;
    ImVec2 mScreenSquareCenter;
    ImVec2 mScreenSquareMin;
    ImVec2 mScreenSquareMax;

    float mScreenFactor;
    vec_t mRelativeOrigin;

    bool mbUsing{false};
    bool mbMouseOver{false};
    bool mReversed{false}; // reversed projection matrix

    // translation
    vec_t mTranslationPlan;
    vec_t mTranslationPlanOrigin;
    vec_t mMatrixOrigin;
    vec_t mTranslationLastDelta;

    // rotation
    vec_t mRotationVectorSource;
    float mRotationAngle;
    float mRotationAngleOrigin;
    // vec_t mWorldToLocalAxis;

    // scale
    vec_t mScale;
    vec_t mScaleValueOrigin;
    vec_t mScaleLast;
    float mSaveMousePosx;

    // save axis factor when using gizmo
    bool mBelowAxisLimit[3];
    bool mBelowPlaneLimit[3];
    float mAxisFactor[3];

    float mAxisLimit{0.0025};
    float mPlaneLimit{0.02};

    int mCurrentOperation;

    float mX{0};
    float mY{0};
    float mWidth{0};
    float mHeight{0};
    float mXMax{0};
    float mYMax{0};
    float mDisplayRatio{1};
    bool mIsOrthographic{false};

    int mActualID = -1;
    int mEditingID = -1;
    OPERATION mOperation = OPERATION(-1);

    float mGizmoSizeClipSpace = 0.1f;
};

static Context gContext;

static constexpr const char *scaleInfoMask[] = {"X : %5.2f", "Y : %5.2f", "Z : %5.2f", "XYZ : %5.2f"};
static const vec_t directionUnary[3] = {makeVect(1.f, 0.f, 0.f), makeVect(0.f, 1.f, 0.f), makeVect(0.f, 0.f, 1.f)};
static constexpr int translationInfoIndex[] = {0, 0, 0, 1, 0, 0, 2, 0, 0, 1, 2, 0, 0, 2, 0, 0, 1, 0, 0, 1, 2};
static constexpr float quadMin = 0.5f;
static constexpr float quadMax = 0.8f;
static constexpr float quadUV[8] = {quadMin, quadMin, quadMin, quadMax, quadMax, quadMax, quadMax, quadMin};
static constexpr int halfCircleSegmentCount = 64;
static constexpr float snapTension = 0.5f;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
static int GetMoveType(OPERATION op, vec_t *gizmoHitProportion);
static int GetRotateType(OPERATION op);
static int GetScaleType(OPERATION op);

static ImU32 GetColorU32(int idx) {
    IM_ASSERT(idx < COLOR::COUNT);
    return ImGui::ColorConvertFloat4ToU32(gContext.mStyle.Colors[idx]);
}

static ImVec2 worldToPos(const vec_t &worldPos, const matrix_t &mat, ImVec2 position = ImVec2(gContext.mX, gContext.mY), ImVec2 size = ImVec2(gContext.mWidth, gContext.mHeight)) {
    vec_t trans;
    trans.TransformPoint(worldPos, mat);
    trans *= 0.5f / trans.w;
    trans += makeVect(0.5f, 0.5f);
    trans.y = 1.f - trans.y;
    trans.x *= size.x;
    trans.y *= size.y;
    trans.x += position.x;
    trans.y += position.y;
    return {trans.x, trans.y};
}

static void ComputeCameraRay(vec_t &rayOrigin, vec_t &rayDir, ImVec2 position = ImVec2(gContext.mX, gContext.mY), ImVec2 size = ImVec2(gContext.mWidth, gContext.mHeight)) {
    ImGuiIO &io = ImGui::GetIO();

    matrix_t mViewProjInverse;
    mViewProjInverse.Inverse(gContext.mViewMat * gContext.mProjectionMat);

    const float mox = ((io.MousePos.x - position.x) / size.x) * 2.f - 1.f;
    const float moy = (1.f - ((io.MousePos.y - position.y) / size.y)) * 2.f - 1.f;

    const float zNear = gContext.mReversed ? (1.f - FLT_EPSILON) : 0.f;
    const float zFar = gContext.mReversed ? 0.f : (1.f - FLT_EPSILON);

    rayOrigin.Transform(makeVect(mox, moy, zNear, 1.f), mViewProjInverse);
    rayOrigin *= 1.f / rayOrigin.w;
    vec_t rayEnd;
    rayEnd.Transform(makeVect(mox, moy, zFar, 1.f), mViewProjInverse);
    rayEnd *= 1.f / rayEnd.w;
    rayDir = Normalized(rayEnd - rayOrigin);
}

static float GetSegmentLengthClipSpace(const vec_t &start, const vec_t &end, const bool localCoordinates = false) {
    vec_t startOfSegment = start;
    const matrix_t &mvp = localCoordinates ? gContext.mMVPLocal : gContext.mMVP;
    startOfSegment.TransformPoint(mvp);
    if (fabsf(startOfSegment.w) > FLT_EPSILON) // check for axis aligned with camera direction
    {
        startOfSegment *= 1.f / startOfSegment.w;
    }

    vec_t endOfSegment = end;
    endOfSegment.TransformPoint(mvp);
    if (fabsf(endOfSegment.w) > FLT_EPSILON) // check for axis aligned with camera direction
    {
        endOfSegment *= 1.f / endOfSegment.w;
    }

    vec_t clipSpaceAxis = endOfSegment - startOfSegment;
    if (gContext.mDisplayRatio < 1.0)
        clipSpaceAxis.x *= gContext.mDisplayRatio;
    else
        clipSpaceAxis.y /= gContext.mDisplayRatio;
    float segmentLengthInClipSpace = sqrtf(clipSpaceAxis.x * clipSpaceAxis.x + clipSpaceAxis.y * clipSpaceAxis.y);
    return segmentLengthInClipSpace;
}

static float GetParallelogram(const vec_t &ptO, const vec_t &ptA, const vec_t &ptB) {
    vec_t pts[] = {ptO, ptA, ptB};
    for (unsigned int i = 0; i < 3; i++) {
        pts[i].TransformPoint(gContext.mMVP);
        if (fabsf(pts[i].w) > FLT_EPSILON) // check for axis aligned with camera direction
        {
            pts[i] *= 1.f / pts[i].w;
        }
    }
    vec_t segA = pts[1] - pts[0];
    vec_t segB = pts[2] - pts[0];
    segA.y /= gContext.mDisplayRatio;
    segB.y /= gContext.mDisplayRatio;
    vec_t segAOrtho = makeVect(-segA.y, segA.x);
    segAOrtho.Normalize();
    float dt = segAOrtho.Dot3(segB);
    float surface = sqrtf(segA.x * segA.x + segA.y * segA.y) * fabsf(dt);
    return surface;
}

inline vec_t PointOnSegment(const vec_t &point, const vec_t &vertPos1, const vec_t &vertPos2) {
    vec_t c = point - vertPos1;
    vec_t V;

    V.Normalize(vertPos2 - vertPos1);
    float d = (vertPos2 - vertPos1).Length();
    float t = V.Dot3(c);

    if (t < 0.f) {
        return vertPos1;
    }

    if (t > d) {
        return vertPos2;
    }

    return vertPos1 + V * t;
}

static float IntersectRayPlane(const vec_t &rOrigin, const vec_t &rVector, const vec_t &plan) {
    const float numer = plan.Dot3(rOrigin) - plan.w;
    const float denom = plan.Dot3(rVector);

    if (fabsf(denom) < FLT_EPSILON) // normal is orthogonal to vector, cant intersect
    {
        return -1.0f;
    }

    return -(numer / denom);
}

static bool IsHoveringWindow() {
    ImGuiContext &g = *ImGui::GetCurrentContext();
    ImGuiWindow *window = ImGui::FindWindowByName(ImGui::GetWindowDrawList()->_OwnerName);
    if (g.HoveredWindow == window) // Mouse hovering drawlist window
        return true;
    if (g.HoveredWindow != NULL) // Any other window is hovered
        return false;
    if (ImGui::IsMouseHoveringRect(window->InnerRect.Min, window->InnerRect.Max, false)) // Hovering drawlist window rect, while no other window is hovered (for _NoInputs windows)
        return true;
    return false;
}

void SetRect(float x, float y, float width, float height) {
    gContext.mX = x;
    gContext.mY = y;
    gContext.mWidth = width;
    gContext.mHeight = height;
    gContext.mXMax = gContext.mX + gContext.mWidth;
    gContext.mYMax = gContext.mY + gContext.mXMax;
    gContext.mDisplayRatio = width / height;
}

bool IsUsing() {
    return (gContext.mbUsing && (gContext.mActualID == -1 || gContext.mActualID == gContext.mEditingID));
}

bool IsOver() {
    return (Intersects(gContext.mOperation, TRANSLATE) && GetMoveType(gContext.mOperation, NULL) != MT_NONE) ||
        (Intersects(gContext.mOperation, ROTATE) && GetRotateType(gContext.mOperation) != MT_NONE) ||
        (Intersects(gContext.mOperation, SCALE) && GetScaleType(gContext.mOperation) != MT_NONE) || IsUsing();
}

bool IsOver(OPERATION op) {
    if (IsUsing()) {
        return true;
    }
    if (Intersects(op, SCALE) && GetScaleType(op) != MT_NONE) {
        return true;
    }
    if (Intersects(op, ROTATE) && GetRotateType(op) != MT_NONE) {
        return true;
    }
    if (Intersects(op, TRANSLATE) && GetMoveType(op, NULL) != MT_NONE) {
        return true;
    }
    return false;
}

static void ComputeContext(const float *view, const float *projection, float *matrix, MODE mode) {
    gContext.mMode = mode;
    gContext.mViewMat = *(matrix_t *)view;
    gContext.mProjectionMat = *(matrix_t *)projection;
    gContext.mbMouseOver = IsHoveringWindow();

    gContext.mModelLocal = *(matrix_t *)matrix;
    gContext.mModelLocal.OrthoNormalize();

    if (mode == LOCAL) {
        gContext.mModel = gContext.mModelLocal;
    } else {
        gContext.mModel.Translation(((matrix_t *)matrix)->v.position);
    }
    gContext.mModelSource = *(matrix_t *)matrix;
    gContext.mModelScaleOrigin.Set(gContext.mModelSource.v.right.Length(), gContext.mModelSource.v.up.Length(), gContext.mModelSource.v.dir.Length());

    gContext.mModelInverse.Inverse(gContext.mModel);
    gContext.mViewProjection = gContext.mViewMat * gContext.mProjectionMat;
    gContext.mMVP = gContext.mModel * gContext.mViewProjection;
    gContext.mMVPLocal = gContext.mModelLocal * gContext.mViewProjection;

    matrix_t viewInverse;
    viewInverse.Inverse(gContext.mViewMat);
    gContext.mCameraDir = viewInverse.v.dir;
    gContext.mCameraEye = viewInverse.v.position;

    // projection reverse
    vec_t nearPos, farPos;
    nearPos.Transform(makeVect(0, 0, 1.f, 1.f), gContext.mProjectionMat);
    farPos.Transform(makeVect(0, 0, 2.f, 1.f), gContext.mProjectionMat);

    gContext.mReversed = (nearPos.z / nearPos.w) > (farPos.z / farPos.w);

    // compute scale from the size of camera right vector projected on screen at the matrix position
    vec_t pointRight = viewInverse.v.right;
    pointRight.TransformPoint(gContext.mViewProjection);
    gContext.mScreenFactor = gContext.mGizmoSizeClipSpace / (pointRight.x / pointRight.w - gContext.mMVP.v.position.x / gContext.mMVP.v.position.w);

    vec_t rightViewInverse = viewInverse.v.right;
    rightViewInverse.TransformVector(gContext.mModelInverse);
    float rightLength = GetSegmentLengthClipSpace(makeVect(0.f, 0.f), rightViewInverse);
    gContext.mScreenFactor = gContext.mGizmoSizeClipSpace / rightLength;

    ImVec2 centerSSpace = worldToPos(makeVect(0.f, 0.f), gContext.mMVP);
    gContext.mScreenSquareCenter = centerSSpace;
    gContext.mScreenSquareMin = ImVec2(centerSSpace.x - 10.f, centerSSpace.y - 10.f);
    gContext.mScreenSquareMax = ImVec2(centerSSpace.x + 10.f, centerSSpace.y + 10.f);

    ComputeCameraRay(gContext.mRayOrigin, gContext.mRayVector);
}

static void ComputeColors(ImU32 *colors, int type, OPERATION operation) {
    ImU32 selectionColor = GetColorU32(SELECTION);

    switch (operation) {
        case TRANSLATE:
            colors[0] = (type == MT_MOVE_SCREEN) ? selectionColor : IM_COL32_WHITE;
            for (int i = 0; i < 3; i++) {
                colors[i + 1] = (type == (int)(MT_MOVE_X + i)) ? selectionColor : GetColorU32(DIRECTION_X + i);
                colors[i + 4] = (type == (int)(MT_MOVE_YZ + i)) ? selectionColor : GetColorU32(PLANE_X + i);
                colors[i + 4] = (type == MT_MOVE_SCREEN) ? selectionColor : colors[i + 4];
            }
            break;
        case ROTATE:
            colors[0] = (type == MT_ROTATE_SCREEN) ? selectionColor : IM_COL32_WHITE;
            for (int i = 0; i < 3; i++) {
                colors[i + 1] = (type == (int)(MT_ROTATE_X + i)) ? selectionColor : GetColorU32(DIRECTION_X + i);
            }
            break;
        case SCALEU:
        case SCALE:
            colors[0] = (type == MT_SCALE_XYZ) ? selectionColor : IM_COL32_WHITE;
            for (int i = 0; i < 3; i++) {
                colors[i + 1] = (type == (int)(MT_SCALE_X + i)) ? selectionColor : GetColorU32(DIRECTION_X + i);
            }
            break;
        // note: this internal function is only called with three possible values for operation
        default:
            break;
    }
}

static void ComputeTripodAxisAndVisibility(const int axisIndex, vec_t &dirAxis, vec_t &dirPlaneX, vec_t &dirPlaneY, bool &belowAxisLimit, bool &belowPlaneLimit, const bool localCoordinates = false) {
    dirAxis = directionUnary[axisIndex];
    dirPlaneX = directionUnary[(axisIndex + 1) % 3];
    dirPlaneY = directionUnary[(axisIndex + 2) % 3];

    if (gContext.mbUsing && (gContext.mActualID == -1 || gContext.mActualID == gContext.mEditingID)) {
        // when using, use stored factors so the gizmo doesn't flip when we translate
        belowAxisLimit = gContext.mBelowAxisLimit[axisIndex];
        belowPlaneLimit = gContext.mBelowPlaneLimit[axisIndex];

        dirAxis *= gContext.mAxisFactor[axisIndex];
        dirPlaneX *= gContext.mAxisFactor[(axisIndex + 1) % 3];
        dirPlaneY *= gContext.mAxisFactor[(axisIndex + 2) % 3];
    } else {
        // new method
        float lenDir = GetSegmentLengthClipSpace(makeVect(0.f, 0.f, 0.f), dirAxis, localCoordinates);
        float lenDirMinus = GetSegmentLengthClipSpace(makeVect(0.f, 0.f, 0.f), -dirAxis, localCoordinates);

        float lenDirPlaneX = GetSegmentLengthClipSpace(makeVect(0.f, 0.f, 0.f), dirPlaneX, localCoordinates);
        float lenDirMinusPlaneX = GetSegmentLengthClipSpace(makeVect(0.f, 0.f, 0.f), -dirPlaneX, localCoordinates);

        float lenDirPlaneY = GetSegmentLengthClipSpace(makeVect(0.f, 0.f, 0.f), dirPlaneY, localCoordinates);
        float lenDirMinusPlaneY = GetSegmentLengthClipSpace(makeVect(0.f, 0.f, 0.f), -dirPlaneY, localCoordinates);

        // For readability, flip gizmo axis for better visibility
        // When false, they always stay along the positive world/local axis
        bool allowFlip = true;
        float mulAxis = (allowFlip && lenDir < lenDirMinus && fabsf(lenDir - lenDirMinus) > FLT_EPSILON) ? -1.f : 1.f;
        float mulAxisX = (allowFlip && lenDirPlaneX < lenDirMinusPlaneX && fabsf(lenDirPlaneX - lenDirMinusPlaneX) > FLT_EPSILON) ? -1.f : 1.f;
        float mulAxisY = (allowFlip && lenDirPlaneY < lenDirMinusPlaneY && fabsf(lenDirPlaneY - lenDirMinusPlaneY) > FLT_EPSILON) ? -1.f : 1.f;
        dirAxis *= mulAxis;
        dirPlaneX *= mulAxisX;
        dirPlaneY *= mulAxisY;

        // for axis
        float axisLengthInClipSpace = GetSegmentLengthClipSpace(makeVect(0.f, 0.f, 0.f), dirAxis * gContext.mScreenFactor, localCoordinates);

        float paraSurf = GetParallelogram(makeVect(0.f, 0.f, 0.f), dirPlaneX * gContext.mScreenFactor, dirPlaneY * gContext.mScreenFactor);
        belowPlaneLimit = (paraSurf > gContext.mAxisLimit);
        belowAxisLimit = (axisLengthInClipSpace > gContext.mPlaneLimit);

        // and store values
        gContext.mAxisFactor[axisIndex] = mulAxis;
        gContext.mAxisFactor[(axisIndex + 1) % 3] = mulAxisX;
        gContext.mAxisFactor[(axisIndex + 2) % 3] = mulAxisY;
        gContext.mBelowAxisLimit[axisIndex] = belowAxisLimit;
        gContext.mBelowPlaneLimit[axisIndex] = belowPlaneLimit;
    }
}

static void ComputeSnap(float *value, float snap) {
    if (snap <= FLT_EPSILON) {
        return;
    }

    float modulo = fmodf(*value, snap);
    float moduloRatio = fabsf(modulo) / snap;
    if (moduloRatio < snapTension) {
        *value -= modulo;
    } else if (moduloRatio > (1.f - snapTension)) {
        *value = *value - modulo + snap * ((*value < 0.f) ? -1.f : 1.f);
    }
}
static void ComputeSnap(vec_t &value, const float *snap) {
    for (int i = 0; i < 3; i++) {
        ComputeSnap(&value[i], snap[i]);
    }
}

static float ComputeAngleOnPlan() {
    const float len = IntersectRayPlane(gContext.mRayOrigin, gContext.mRayVector, gContext.mTranslationPlan);
    vec_t localPos = Normalized(gContext.mRayOrigin + gContext.mRayVector * len - gContext.mModel.v.position);

    vec_t perpendicularVector;
    perpendicularVector.Cross(gContext.mRotationVectorSource, gContext.mTranslationPlan);
    perpendicularVector.Normalize();
    float acosAngle = Clamp(Dot(localPos, gContext.mRotationVectorSource), -1.f, 1.f);
    float angle = acosf(acosAngle);
    angle *= (Dot(localPos, perpendicularVector) < 0.f) ? 1.f : -1.f;
    return angle;
}

static void DrawRotationGizmo(OPERATION op, int type) {
    if (!Intersects(op, ROTATE)) {
        return;
    }
    ImDrawList *drawList = ImGui::GetWindowDrawList();

    // colors
    ImU32 colors[7];
    ComputeColors(colors, type, ROTATE);

    vec_t cameraToModelNormalized;
    if (gContext.mIsOrthographic) {
        matrix_t viewInverse;
        viewInverse.Inverse(*(matrix_t *)&gContext.mViewMat);
        cameraToModelNormalized = -viewInverse.v.dir;
    } else {
        cameraToModelNormalized = Normalized(gContext.mModel.v.position - gContext.mCameraEye);
    }

    cameraToModelNormalized.TransformVector(gContext.mModelInverse);

    gContext.mRadiusSquareCenter = screenRotateSize * gContext.mHeight;

    bool hasRSC = Intersects(op, ROTATE_SCREEN);
    for (int axis = 0; axis < 3; axis++) {
        if (!Intersects(op, static_cast<OPERATION>(ROTATE_Z >> axis))) {
            continue;
        }
        const bool usingAxis = (gContext.mbUsing && type == MT_ROTATE_Z - axis);
        const int circleMul = (hasRSC && !usingAxis) ? 1 : 2;

        ImVec2 *circlePos = (ImVec2 *)alloca(sizeof(ImVec2) * (circleMul * halfCircleSegmentCount + 1));

        float angleStart = atan2f(cameraToModelNormalized[(4 - axis) % 3], cameraToModelNormalized[(3 - axis) % 3]) + ZPI * 0.5f;

        for (int i = 0; i < circleMul * halfCircleSegmentCount + 1; i++) {
            float ng = angleStart + (float)circleMul * ZPI * ((float)i / (float)(circleMul * halfCircleSegmentCount));
            vec_t axisPos = makeVect(cosf(ng), sinf(ng), 0.f);
            vec_t pos = makeVect(axisPos[axis], axisPos[(axis + 1) % 3], axisPos[(axis + 2) % 3]) * gContext.mScreenFactor * rotationDisplayFactor;
            circlePos[i] = worldToPos(pos, gContext.mMVP);
        }
        if (!gContext.mbUsing || usingAxis) {
            drawList->AddPolyline(circlePos, circleMul * halfCircleSegmentCount + 1, colors[3 - axis], false, gContext.mStyle.RotationLineThickness);
        }

        float radiusAxis = sqrtf((ImLengthSqr(worldToPos(gContext.mModel.v.position, gContext.mViewProjection) - circlePos[0])));
        if (radiusAxis > gContext.mRadiusSquareCenter) {
            gContext.mRadiusSquareCenter = radiusAxis;
        }
    }
    if (hasRSC && (!gContext.mbUsing || type == MT_ROTATE_SCREEN)) {
        drawList->AddCircle(worldToPos(gContext.mModel.v.position, gContext.mViewProjection), gContext.mRadiusSquareCenter, colors[0], 64, gContext.mStyle.RotationOuterLineThickness);
    }

    if (gContext.mbUsing && (gContext.mActualID == -1 || gContext.mActualID == gContext.mEditingID) && IsRotateType(type)) {
        ImVec2 circlePos[halfCircleSegmentCount + 1];

        circlePos[0] = worldToPos(gContext.mModel.v.position, gContext.mViewProjection);
        for (unsigned int i = 1; i < halfCircleSegmentCount + 1; i++) {
            float ng = gContext.mRotationAngle * ((float)(i - 1) / (float)(halfCircleSegmentCount - 1));
            matrix_t rotateVectorMatrix;
            rotateVectorMatrix.RotationAxis(gContext.mTranslationPlan, ng);
            vec_t pos;
            pos.TransformPoint(gContext.mRotationVectorSource, rotateVectorMatrix);
            pos *= gContext.mScreenFactor * rotationDisplayFactor;
            circlePos[i] = worldToPos(pos + gContext.mModel.v.position, gContext.mViewProjection);
        }
        drawList->AddConvexPolyFilled(circlePos, halfCircleSegmentCount + 1, GetColorU32(ROTATION_USING_FILL));
        drawList->AddPolyline(circlePos, halfCircleSegmentCount + 1, GetColorU32(ROTATION_USING_BORDER), true, gContext.mStyle.RotationLineThickness);

        ImVec2 destinationPosOnScreen = circlePos[1];
        char tmps[512];
        static constexpr const char *rotationInfoMask[] = {"X : %5.2f deg %5.2f rad", "Y : %5.2f deg %5.2f rad", "Z : %5.2f deg %5.2f rad", "Screen : %5.2f deg %5.2f rad"};
        ImFormatString(tmps, sizeof(tmps), rotationInfoMask[type - MT_ROTATE_X], (gContext.mRotationAngle / ZPI) * 180.f, gContext.mRotationAngle);
        drawList->AddText(ImVec2(destinationPosOnScreen.x + 15, destinationPosOnScreen.y + 15), GetColorU32(TEXT_SHADOW), tmps);
        drawList->AddText(ImVec2(destinationPosOnScreen.x + 14, destinationPosOnScreen.y + 14), GetColorU32(TEXT), tmps);
    }
}

static void DrawHatchedAxis(const vec_t &axis) {
    if (gContext.mStyle.HatchedAxisLineThickness <= 0.0f) {
        return;
    }

    for (int j = 1; j < 10; j++) {
        ImVec2 baseSSpace2 = worldToPos(axis * 0.05f * (float)(j * 2) * gContext.mScreenFactor, gContext.mMVP);
        ImVec2 worldDirSSpace2 = worldToPos(axis * 0.05f * (float)(j * 2 + 1) * gContext.mScreenFactor, gContext.mMVP);
        ImGui::GetWindowDrawList()->AddLine(baseSSpace2, worldDirSSpace2, GetColorU32(HATCHED_AXIS_LINES), gContext.mStyle.HatchedAxisLineThickness);
    }
}

static void DrawScaleGizmo(OPERATION op, int type) {
    ImDrawList *drawList = ImGui::GetWindowDrawList();

    if (!Intersects(op, SCALE)) {
        return;
    }

    // colors
    ImU32 colors[7];
    ComputeColors(colors, type, SCALE);

    // draw
    vec_t scaleDisplay = {1.f, 1.f, 1.f, 1.f};

    if (gContext.mbUsing && (gContext.mActualID == -1 || gContext.mActualID == gContext.mEditingID)) {
        scaleDisplay = gContext.mScale;
    }

    for (int i = 0; i < 3; i++) {
        if (!Intersects(op, static_cast<OPERATION>(SCALE_X << i))) {
            continue;
        }
        const bool usingAxis = (gContext.mbUsing && type == MT_SCALE_X + i);
        if (!gContext.mbUsing || usingAxis) {
            vec_t dirPlaneX, dirPlaneY, dirAxis;
            bool belowAxisLimit, belowPlaneLimit;
            ComputeTripodAxisAndVisibility(i, dirAxis, dirPlaneX, dirPlaneY, belowAxisLimit, belowPlaneLimit, true);

            // draw axis
            if (belowAxisLimit) {
                bool hasTranslateOnAxis = Contains(op, static_cast<OPERATION>(TRANSLATE_X << i));
                float markerScale = hasTranslateOnAxis ? 1.4f : 1.0f;
                ImVec2 baseSSpace = worldToPos(dirAxis * 0.1f * gContext.mScreenFactor, gContext.mMVP);
                ImVec2 worldDirSSpaceNoScale = worldToPos(dirAxis * markerScale * gContext.mScreenFactor, gContext.mMVP);
                ImVec2 worldDirSSpace = worldToPos((dirAxis * markerScale * scaleDisplay[i]) * gContext.mScreenFactor, gContext.mMVP);

                if (gContext.mbUsing && (gContext.mActualID == -1 || gContext.mActualID == gContext.mEditingID)) {
                    ImU32 scaleLineColor = GetColorU32(SCALE_LINE);
                    drawList->AddLine(baseSSpace, worldDirSSpaceNoScale, scaleLineColor, gContext.mStyle.ScaleLineThickness);
                    drawList->AddCircleFilled(worldDirSSpaceNoScale, gContext.mStyle.ScaleLineCircleSize, scaleLineColor);
                }

                if (!hasTranslateOnAxis || gContext.mbUsing) {
                    drawList->AddLine(baseSSpace, worldDirSSpace, colors[i + 1], gContext.mStyle.ScaleLineThickness);
                }
                drawList->AddCircleFilled(worldDirSSpace, gContext.mStyle.ScaleLineCircleSize, colors[i + 1]);

                if (gContext.mAxisFactor[i] < 0.f) {
                    DrawHatchedAxis(dirAxis * scaleDisplay[i]);
                }
            }
        }
    }

    // draw screen cirle
    drawList->AddCircleFilled(gContext.mScreenSquareCenter, gContext.mStyle.CenterCircleSize, colors[0], 32);

    if (gContext.mbUsing && (gContext.mActualID == -1 || gContext.mActualID == gContext.mEditingID) && IsScaleType(type)) {
        ImVec2 destinationPosOnScreen = worldToPos(gContext.mModel.v.position, gContext.mViewProjection);
        char tmps[512];
        // vec_t deltaInfo = gContext.mModel.v.position - gContext.mMatrixOrigin;
        int componentInfoIndex = (type - MT_SCALE_X) * 3;
        ImFormatString(tmps, sizeof(tmps), scaleInfoMask[type - MT_SCALE_X], scaleDisplay[translationInfoIndex[componentInfoIndex]]);
        drawList->AddText(ImVec2(destinationPosOnScreen.x + 15, destinationPosOnScreen.y + 15), GetColorU32(TEXT_SHADOW), tmps);
        drawList->AddText(ImVec2(destinationPosOnScreen.x + 14, destinationPosOnScreen.y + 14), GetColorU32(TEXT), tmps);
    }
}

static void DrawScaleUniveralGizmo(OPERATION op, int type) {
    ImDrawList *drawList = ImGui::GetWindowDrawList();

    if (!Intersects(op, SCALEU)) {
        return;
    }

    // colors
    ImU32 colors[7];
    ComputeColors(colors, type, SCALEU);

    // draw
    vec_t scaleDisplay = {1.f, 1.f, 1.f, 1.f};

    if (gContext.mbUsing && (gContext.mActualID == -1 || gContext.mActualID == gContext.mEditingID)) {
        scaleDisplay = gContext.mScale;
    }

    for (int i = 0; i < 3; i++) {
        if (!Intersects(op, static_cast<OPERATION>(SCALE_XU << i))) {
            continue;
        }
        const bool usingAxis = (gContext.mbUsing && type == MT_SCALE_X + i);
        if (!gContext.mbUsing || usingAxis) {
            vec_t dirPlaneX, dirPlaneY, dirAxis;
            bool belowAxisLimit, belowPlaneLimit;
            ComputeTripodAxisAndVisibility(i, dirAxis, dirPlaneX, dirPlaneY, belowAxisLimit, belowPlaneLimit, true);

            // draw axis
            if (belowAxisLimit) {
                bool hasTranslateOnAxis = Contains(op, static_cast<OPERATION>(TRANSLATE_X << i));
                float markerScale = hasTranslateOnAxis ? 1.4f : 1.0f;
                // ImVec2 baseSSpace = worldToPos(dirAxis * 0.1f * gContext.mScreenFactor, gContext.mMVPLocal);
                // ImVec2 worldDirSSpaceNoScale = worldToPos(dirAxis * markerScale * gContext.mScreenFactor, gContext.mMVP);
                ImVec2 worldDirSSpace = worldToPos((dirAxis * markerScale * scaleDisplay[i]) * gContext.mScreenFactor, gContext.mMVPLocal);

                drawList->AddCircleFilled(worldDirSSpace, 12.f, colors[i + 1]);
            }
        }
    }

    // draw screen cirle
    drawList->AddCircle(gContext.mScreenSquareCenter, 20.f, colors[0], 32, gContext.mStyle.CenterCircleSize);

    if (gContext.mbUsing && (gContext.mActualID == -1 || gContext.mActualID == gContext.mEditingID) && IsScaleType(type)) {
        ImVec2 destinationPosOnScreen = worldToPos(gContext.mModel.v.position, gContext.mViewProjection);
        char tmps[512];
        // vec_t deltaInfo = gContext.mModel.v.position - gContext.mMatrixOrigin;
        int componentInfoIndex = (type - MT_SCALE_X) * 3;
        ImFormatString(tmps, sizeof(tmps), scaleInfoMask[type - MT_SCALE_X], scaleDisplay[translationInfoIndex[componentInfoIndex]]);
        drawList->AddText(ImVec2(destinationPosOnScreen.x + 15, destinationPosOnScreen.y + 15), GetColorU32(TEXT_SHADOW), tmps);
        drawList->AddText(ImVec2(destinationPosOnScreen.x + 14, destinationPosOnScreen.y + 14), GetColorU32(TEXT), tmps);
    }
}

static void DrawTranslationGizmo(OPERATION op, int type) {
    ImDrawList *drawList = ImGui::GetWindowDrawList();
    if (!drawList) {
        return;
    }

    if (!Intersects(op, TRANSLATE)) {
        return;
    }

    // colors
    ImU32 colors[7];
    ComputeColors(colors, type, TRANSLATE);

    const ImVec2 origin = worldToPos(gContext.mModel.v.position, gContext.mViewProjection);

    // draw
    bool belowAxisLimit = false;
    bool belowPlaneLimit = false;
    for (int i = 0; i < 3; ++i) {
        vec_t dirPlaneX, dirPlaneY, dirAxis;
        ComputeTripodAxisAndVisibility(i, dirAxis, dirPlaneX, dirPlaneY, belowAxisLimit, belowPlaneLimit);

        if (!gContext.mbUsing || (gContext.mbUsing && type == MT_MOVE_X + i)) {
            // draw axis
            if (belowAxisLimit && Intersects(op, static_cast<OPERATION>(TRANSLATE_X << i))) {
                ImVec2 baseSSpace = worldToPos(dirAxis * 0.1f * gContext.mScreenFactor, gContext.mMVP);
                ImVec2 worldDirSSpace = worldToPos(dirAxis * gContext.mScreenFactor, gContext.mMVP);

                drawList->AddLine(baseSSpace, worldDirSSpace, colors[i + 1], gContext.mStyle.TranslationLineThickness);

                // Arrow head begin
                ImVec2 dir(origin - worldDirSSpace);

                float d = sqrtf(ImLengthSqr(dir));
                dir /= d; // Normalize
                dir *= gContext.mStyle.TranslationLineArrowSize;

                ImVec2 ortogonalDir(dir.y, -dir.x); // Perpendicular vector
                ImVec2 a(worldDirSSpace + dir);
                drawList->AddTriangleFilled(worldDirSSpace - dir, a + ortogonalDir, a - ortogonalDir, colors[i + 1]);
                // Arrow head end

                if (gContext.mAxisFactor[i] < 0.f) {
                    DrawHatchedAxis(dirAxis);
                }
            }
        }
        // draw plane
        if (!gContext.mbUsing || (gContext.mbUsing && type == MT_MOVE_YZ + i)) {
            if (belowPlaneLimit && Contains(op, TRANSLATE_PLANS[i])) {
                ImVec2 screenQuadPts[4];
                for (int j = 0; j < 4; ++j) {
                    vec_t cornerWorldPos = (dirPlaneX * quadUV[j * 2] + dirPlaneY * quadUV[j * 2 + 1]) * gContext.mScreenFactor;
                    screenQuadPts[j] = worldToPos(cornerWorldPos, gContext.mMVP);
                }
                drawList->AddPolyline(screenQuadPts, 4, GetColorU32(DIRECTION_X + i), true, 1.0f);
                drawList->AddConvexPolyFilled(screenQuadPts, 4, colors[i + 4]);
            }
        }
    }

    drawList->AddCircleFilled(gContext.mScreenSquareCenter, gContext.mStyle.CenterCircleSize, colors[0], 32);

    if (gContext.mbUsing && (gContext.mActualID == -1 || gContext.mActualID == gContext.mEditingID) && IsTranslateType(type)) {
        ImU32 translationLineColor = GetColorU32(TRANSLATION_LINE);

        ImVec2 sourcePosOnScreen = worldToPos(gContext.mMatrixOrigin, gContext.mViewProjection);
        ImVec2 destinationPosOnScreen = worldToPos(gContext.mModel.v.position, gContext.mViewProjection);
        vec_t dif = {destinationPosOnScreen.x - sourcePosOnScreen.x, destinationPosOnScreen.y - sourcePosOnScreen.y, 0.f, 0.f};
        dif.Normalize();
        dif *= 5.f;
        drawList->AddCircle(sourcePosOnScreen, 6.f, translationLineColor);
        drawList->AddCircle(destinationPosOnScreen, 6.f, translationLineColor);
        drawList->AddLine(ImVec2(sourcePosOnScreen.x + dif.x, sourcePosOnScreen.y + dif.y), ImVec2(destinationPosOnScreen.x - dif.x, destinationPosOnScreen.y - dif.y), translationLineColor, 2.f);

        char tmps[512];
        vec_t deltaInfo = gContext.mModel.v.position - gContext.mMatrixOrigin;
        int componentInfoIndex = (type - MT_MOVE_X) * 3;
        static constexpr const char *translationInfoMask[] = {"X : %5.3f", "Y : %5.3f", "Z : %5.3f", "Y : %5.3f Z : %5.3f", "X : %5.3f Z : %5.3f", "X : %5.3f Y : %5.3f", "X : %5.3f Y : %5.3f Z : %5.3f"};
        ImFormatString(tmps, sizeof(tmps), translationInfoMask[type - MT_MOVE_X], deltaInfo[translationInfoIndex[componentInfoIndex]], deltaInfo[translationInfoIndex[componentInfoIndex + 1]], deltaInfo[translationInfoIndex[componentInfoIndex + 2]]);
        drawList->AddText(ImVec2(destinationPosOnScreen.x + 15, destinationPosOnScreen.y + 15), GetColorU32(TEXT_SHADOW), tmps);
        drawList->AddText(ImVec2(destinationPosOnScreen.x + 14, destinationPosOnScreen.y + 14), GetColorU32(TEXT), tmps);
    }
}

static bool CanActivate() {
    if (ImGui::IsMouseClicked(0) && !ImGui::IsAnyItemHovered() && !ImGui::IsAnyItemActive()) {
        return true;
    }
    return false;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//

static int GetScaleType(OPERATION op) {
    if (gContext.mbUsing) {
        return MT_NONE;
    }
    ImGuiIO &io = ImGui::GetIO();
    int type = MT_NONE;

    // screen
    if (io.MousePos.x >= gContext.mScreenSquareMin.x && io.MousePos.x <= gContext.mScreenSquareMax.x &&
        io.MousePos.y >= gContext.mScreenSquareMin.y && io.MousePos.y <= gContext.mScreenSquareMax.y &&
        Contains(op, SCALE)) {
        type = MT_SCALE_XYZ;
    }

    // compute
    for (int i = 0; i < 3 && type == MT_NONE; i++) {
        if (!Intersects(op, static_cast<OPERATION>(SCALE_X << i))) {
            continue;
        }
        vec_t dirPlaneX, dirPlaneY, dirAxis;
        bool belowAxisLimit, belowPlaneLimit;
        ComputeTripodAxisAndVisibility(i, dirAxis, dirPlaneX, dirPlaneY, belowAxisLimit, belowPlaneLimit, true);
        dirAxis.TransformVector(gContext.mModelLocal);
        dirPlaneX.TransformVector(gContext.mModelLocal);
        dirPlaneY.TransformVector(gContext.mModelLocal);

        const float len = IntersectRayPlane(gContext.mRayOrigin, gContext.mRayVector, BuildPlan(gContext.mModelLocal.v.position, dirAxis));
        vec_t posOnPlan = gContext.mRayOrigin + gContext.mRayVector * len;

        const float startOffset = Contains(op, static_cast<OPERATION>(TRANSLATE_X << i)) ? 1.0f : 0.1f;
        const float endOffset = Contains(op, static_cast<OPERATION>(TRANSLATE_X << i)) ? 1.4f : 1.0f;
        const ImVec2 posOnPlanScreen = worldToPos(posOnPlan, gContext.mViewProjection);
        const ImVec2 axisStartOnScreen = worldToPos(gContext.mModelLocal.v.position + dirAxis * gContext.mScreenFactor * startOffset, gContext.mViewProjection);
        const ImVec2 axisEndOnScreen = worldToPos(gContext.mModelLocal.v.position + dirAxis * gContext.mScreenFactor * endOffset, gContext.mViewProjection);

        vec_t closestPointOnAxis = PointOnSegment(makeVect(posOnPlanScreen), makeVect(axisStartOnScreen), makeVect(axisEndOnScreen));

        if ((closestPointOnAxis - makeVect(posOnPlanScreen)).Length() < 12.f) // pixel size
        {
            type = MT_SCALE_X + i;
        }
    }

    // universal

    vec_t deltaScreen = {io.MousePos.x - gContext.mScreenSquareCenter.x, io.MousePos.y - gContext.mScreenSquareCenter.y, 0.f, 0.f};
    float dist = deltaScreen.Length();
    if (Contains(op, SCALEU) && dist >= 17.0f && dist < 23.0f) {
        type = MT_SCALE_XYZ;
    }

    for (int i = 0; i < 3 && type == MT_NONE; i++) {
        if (!Intersects(op, static_cast<OPERATION>(SCALE_XU << i))) {
            continue;
        }

        vec_t dirPlaneX, dirPlaneY, dirAxis;
        bool belowAxisLimit, belowPlaneLimit;
        ComputeTripodAxisAndVisibility(i, dirAxis, dirPlaneX, dirPlaneY, belowAxisLimit, belowPlaneLimit, true);

        // draw axis
        if (belowAxisLimit) {
            bool hasTranslateOnAxis = Contains(op, static_cast<OPERATION>(TRANSLATE_X << i));
            float markerScale = hasTranslateOnAxis ? 1.4f : 1.0f;
            // ImVec2 baseSSpace = worldToPos(dirAxis * 0.1f * gContext.mScreenFactor, gContext.mMVPLocal);
            // ImVec2 worldDirSSpaceNoScale = worldToPos(dirAxis * markerScale * gContext.mScreenFactor, gContext.mMVP);
            ImVec2 worldDirSSpace = worldToPos((dirAxis * markerScale) * gContext.mScreenFactor, gContext.mMVPLocal);

            float distance = sqrtf(ImLengthSqr(worldDirSSpace - io.MousePos));
            if (distance < 12.f) {
                type = MT_SCALE_X + i;
            }
        }
    }
    return type;
}

static int GetRotateType(OPERATION op) {
    if (gContext.mbUsing) {
        return MT_NONE;
    }
    ImGuiIO &io = ImGui::GetIO();
    int type = MT_NONE;

    vec_t deltaScreen = {io.MousePos.x - gContext.mScreenSquareCenter.x, io.MousePos.y - gContext.mScreenSquareCenter.y, 0.f, 0.f};
    float dist = deltaScreen.Length();
    if (Intersects(op, ROTATE_SCREEN) && dist >= (gContext.mRadiusSquareCenter - 4.0f) && dist < (gContext.mRadiusSquareCenter + 4.0f)) {
        type = MT_ROTATE_SCREEN;
    }

    const vec_t planNormals[] = {gContext.mModel.v.right, gContext.mModel.v.up, gContext.mModel.v.dir};

    vec_t modelViewPos;
    modelViewPos.TransformPoint(gContext.mModel.v.position, gContext.mViewMat);

    for (int i = 0; i < 3 && type == MT_NONE; i++) {
        if (!Intersects(op, static_cast<OPERATION>(ROTATE_X << i))) {
            continue;
        }
        // pickup plan
        vec_t pickupPlan = BuildPlan(gContext.mModel.v.position, planNormals[i]);

        const float len = IntersectRayPlane(gContext.mRayOrigin, gContext.mRayVector, pickupPlan);
        const vec_t intersectWorldPos = gContext.mRayOrigin + gContext.mRayVector * len;
        vec_t intersectViewPos;
        intersectViewPos.TransformPoint(intersectWorldPos, gContext.mViewMat);

        if (ImAbs(modelViewPos.z) - ImAbs(intersectViewPos.z) < -FLT_EPSILON) {
            continue;
        }

        const vec_t localPos = intersectWorldPos - gContext.mModel.v.position;
        vec_t idealPosOnCircle = Normalized(localPos);
        idealPosOnCircle.TransformVector(gContext.mModelInverse);
        const ImVec2 idealPosOnCircleScreen = worldToPos(idealPosOnCircle * rotationDisplayFactor * gContext.mScreenFactor, gContext.mMVP);

        // gContext.mDrawList->AddCircle(idealPosOnCircleScreen, 5.f, IM_COL32_WHITE);
        const ImVec2 distanceOnScreen = idealPosOnCircleScreen - io.MousePos;

        const float distance = makeVect(distanceOnScreen).Length();
        if (distance < 8.f) // pixel size
        {
            type = MT_ROTATE_X + i;
        }
    }

    return type;
}

static int GetMoveType(OPERATION op, vec_t *gizmoHitProportion) {
    if (!Intersects(op, TRANSLATE) || gContext.mbUsing || !gContext.mbMouseOver) {
        return MT_NONE;
    }
    ImGuiIO &io = ImGui::GetIO();
    int type = MT_NONE;

    // screen
    if (io.MousePos.x >= gContext.mScreenSquareMin.x && io.MousePos.x <= gContext.mScreenSquareMax.x &&
        io.MousePos.y >= gContext.mScreenSquareMin.y && io.MousePos.y <= gContext.mScreenSquareMax.y &&
        Contains(op, TRANSLATE)) {
        type = MT_MOVE_SCREEN;
    }

    const vec_t screenCoord = makeVect(io.MousePos - ImVec2(gContext.mX, gContext.mY));

    // compute
    for (int i = 0; i < 3 && type == MT_NONE; i++) {
        vec_t dirPlaneX, dirPlaneY, dirAxis;
        bool belowAxisLimit, belowPlaneLimit;
        ComputeTripodAxisAndVisibility(i, dirAxis, dirPlaneX, dirPlaneY, belowAxisLimit, belowPlaneLimit);
        dirAxis.TransformVector(gContext.mModel);
        dirPlaneX.TransformVector(gContext.mModel);
        dirPlaneY.TransformVector(gContext.mModel);

        const float len = IntersectRayPlane(gContext.mRayOrigin, gContext.mRayVector, BuildPlan(gContext.mModel.v.position, dirAxis));
        vec_t posOnPlan = gContext.mRayOrigin + gContext.mRayVector * len;

        const ImVec2 axisStartOnScreen = worldToPos(gContext.mModel.v.position + dirAxis * gContext.mScreenFactor * 0.1f, gContext.mViewProjection) - ImVec2(gContext.mX, gContext.mY);
        const ImVec2 axisEndOnScreen = worldToPos(gContext.mModel.v.position + dirAxis * gContext.mScreenFactor, gContext.mViewProjection) - ImVec2(gContext.mX, gContext.mY);

        vec_t closestPointOnAxis = PointOnSegment(screenCoord, makeVect(axisStartOnScreen), makeVect(axisEndOnScreen));
        if ((closestPointOnAxis - screenCoord).Length() < 12.f && Intersects(op, static_cast<OPERATION>(TRANSLATE_X << i))) // pixel size
        {
            type = MT_MOVE_X + i;
        }

        const float dx = dirPlaneX.Dot3((posOnPlan - gContext.mModel.v.position) * (1.f / gContext.mScreenFactor));
        const float dy = dirPlaneY.Dot3((posOnPlan - gContext.mModel.v.position) * (1.f / gContext.mScreenFactor));
        if (belowPlaneLimit && dx >= quadUV[0] && dx <= quadUV[4] && dy >= quadUV[1] && dy <= quadUV[3] && Contains(op, TRANSLATE_PLANS[i])) {
            type = MT_MOVE_YZ + i;
        }

        if (gizmoHitProportion) {
            *gizmoHitProportion = makeVect(dx, dy, 0.f);
        }
    }
    return type;
}

static bool HandleTranslation(float *matrix, float *deltaMatrix, OPERATION op, int &type, const float *snap) {
    if (!Intersects(op, TRANSLATE) || type != MT_NONE) {
        return false;
    }
    const ImGuiIO &io = ImGui::GetIO();
    const bool applyRotationLocaly = gContext.mMode == LOCAL || type == MT_MOVE_SCREEN;
    bool modified = false;

    // move
    if (gContext.mbUsing && (gContext.mActualID == -1 || gContext.mActualID == gContext.mEditingID) && IsTranslateType(gContext.mCurrentOperation)) {
        ImGui::SetNextFrameWantCaptureMouse(true);
        const float signedLength = IntersectRayPlane(gContext.mRayOrigin, gContext.mRayVector, gContext.mTranslationPlan);
        const float len = fabsf(signedLength); // near plan
        const vec_t newPos = gContext.mRayOrigin + gContext.mRayVector * len;

        // compute delta
        const vec_t newOrigin = newPos - gContext.mRelativeOrigin * gContext.mScreenFactor;
        vec_t delta = newOrigin - gContext.mModel.v.position;

        // 1 axis constraint
        if (gContext.mCurrentOperation >= MT_MOVE_X && gContext.mCurrentOperation <= MT_MOVE_Z) {
            const int axisIndex = gContext.mCurrentOperation - MT_MOVE_X;
            const vec_t &axisValue = *(vec_t *)&gContext.mModel.m[axisIndex];
            const float lengthOnAxis = Dot(axisValue, delta);
            delta = axisValue * lengthOnAxis;
        }

        // snap
        if (snap) {
            vec_t cumulativeDelta = gContext.mModel.v.position + delta - gContext.mMatrixOrigin;
            if (applyRotationLocaly) {
                matrix_t modelSourceNormalized = gContext.mModelSource;
                modelSourceNormalized.OrthoNormalize();
                matrix_t modelSourceNormalizedInverse;
                modelSourceNormalizedInverse.Inverse(modelSourceNormalized);
                cumulativeDelta.TransformVector(modelSourceNormalizedInverse);
                ComputeSnap(cumulativeDelta, snap);
                cumulativeDelta.TransformVector(modelSourceNormalized);
            } else {
                ComputeSnap(cumulativeDelta, snap);
            }
            delta = gContext.mMatrixOrigin + cumulativeDelta - gContext.mModel.v.position;
        }

        if (delta != gContext.mTranslationLastDelta) {
            modified = true;
        }
        gContext.mTranslationLastDelta = delta;

        // compute matrix & delta
        matrix_t deltaMatrixTranslation;
        deltaMatrixTranslation.Translation(delta);
        if (deltaMatrix) {
            memcpy(deltaMatrix, deltaMatrixTranslation.m16, sizeof(float) * 16);
        }

        const matrix_t res = gContext.mModelSource * deltaMatrixTranslation;
        *(matrix_t *)matrix = res;

        if (!io.MouseDown[0]) {
            gContext.mbUsing = false;
        }

        type = gContext.mCurrentOperation;
    } else {
        // find new possible way to move
        vec_t gizmoHitProportion;
        type = GetMoveType(op, &gizmoHitProportion);
        if (type != MT_NONE) {
            ImGui::SetNextFrameWantCaptureMouse(true);
        }
        if (CanActivate() && type != MT_NONE) {
            gContext.mbUsing = true;
            gContext.mEditingID = gContext.mActualID;
            gContext.mCurrentOperation = type;
            vec_t movePlanNormal[] = {gContext.mModel.v.right, gContext.mModel.v.up, gContext.mModel.v.dir, gContext.mModel.v.right, gContext.mModel.v.up, gContext.mModel.v.dir, -gContext.mCameraDir};

            vec_t cameraToModelNormalized = Normalized(gContext.mModel.v.position - gContext.mCameraEye);
            for (unsigned int i = 0; i < 3; i++) {
                vec_t orthoVector = Cross(movePlanNormal[i], cameraToModelNormalized);
                movePlanNormal[i].Cross(orthoVector);
                movePlanNormal[i].Normalize();
            }
            // pickup plan
            gContext.mTranslationPlan = BuildPlan(gContext.mModel.v.position, movePlanNormal[type - MT_MOVE_X]);
            const float len = IntersectRayPlane(gContext.mRayOrigin, gContext.mRayVector, gContext.mTranslationPlan);
            gContext.mTranslationPlanOrigin = gContext.mRayOrigin + gContext.mRayVector * len;
            gContext.mMatrixOrigin = gContext.mModel.v.position;

            gContext.mRelativeOrigin = (gContext.mTranslationPlanOrigin - gContext.mModel.v.position) * (1.f / gContext.mScreenFactor);
        }
    }
    return modified;
}

static bool HandleScale(float *matrix, float *deltaMatrix, OPERATION op, int &type, const float *snap) {
    if ((!Intersects(op, SCALE) && !Intersects(op, SCALEU)) || type != MT_NONE || !gContext.mbMouseOver) {
        return false;
    }
    ImGuiIO &io = ImGui::GetIO();
    bool modified = false;

    if (!gContext.mbUsing) {
        // find new possible way to scale
        type = GetScaleType(op);
        if (type != MT_NONE) {
            ImGui::SetNextFrameWantCaptureMouse(true);
        }
        if (CanActivate() && type != MT_NONE) {
            gContext.mbUsing = true;
            gContext.mEditingID = gContext.mActualID;
            gContext.mCurrentOperation = type;
            const vec_t movePlanNormal[] = {gContext.mModel.v.up, gContext.mModel.v.dir, gContext.mModel.v.right, gContext.mModel.v.dir, gContext.mModel.v.up, gContext.mModel.v.right, -gContext.mCameraDir};
            // pickup plan

            gContext.mTranslationPlan = BuildPlan(gContext.mModel.v.position, movePlanNormal[type - MT_SCALE_X]);
            const float len = IntersectRayPlane(gContext.mRayOrigin, gContext.mRayVector, gContext.mTranslationPlan);
            gContext.mTranslationPlanOrigin = gContext.mRayOrigin + gContext.mRayVector * len;
            gContext.mMatrixOrigin = gContext.mModel.v.position;
            gContext.mScale.Set(1.f, 1.f, 1.f);
            gContext.mRelativeOrigin = (gContext.mTranslationPlanOrigin - gContext.mModel.v.position) * (1.f / gContext.mScreenFactor);
            gContext.mScaleValueOrigin = makeVect(gContext.mModelSource.v.right.Length(), gContext.mModelSource.v.up.Length(), gContext.mModelSource.v.dir.Length());
            gContext.mSaveMousePosx = io.MousePos.x;
        }
    }
    // scale
    if (gContext.mbUsing && (gContext.mActualID == -1 || gContext.mActualID == gContext.mEditingID) && IsScaleType(gContext.mCurrentOperation)) {
        ImGui::SetNextFrameWantCaptureMouse(true);
        const float len = IntersectRayPlane(gContext.mRayOrigin, gContext.mRayVector, gContext.mTranslationPlan);
        vec_t newPos = gContext.mRayOrigin + gContext.mRayVector * len;
        vec_t newOrigin = newPos - gContext.mRelativeOrigin * gContext.mScreenFactor;
        vec_t delta = newOrigin - gContext.mModelLocal.v.position;

        // 1 axis constraint
        if (gContext.mCurrentOperation >= MT_SCALE_X && gContext.mCurrentOperation <= MT_SCALE_Z) {
            int axisIndex = gContext.mCurrentOperation - MT_SCALE_X;
            const vec_t &axisValue = *(vec_t *)&gContext.mModelLocal.m[axisIndex];
            float lengthOnAxis = Dot(axisValue, delta);
            delta = axisValue * lengthOnAxis;

            vec_t baseVector = gContext.mTranslationPlanOrigin - gContext.mModelLocal.v.position;
            float ratio = Dot(axisValue, baseVector + delta) / Dot(axisValue, baseVector);

            gContext.mScale[axisIndex] = max(ratio, 0.001f);
        } else {
            float scaleDelta = (io.MousePos.x - gContext.mSaveMousePosx) * 0.01f;
            gContext.mScale.Set(max(1.f + scaleDelta, 0.001f));
        }

        // snap
        if (snap) {
            float scaleSnap[] = {snap[0], snap[0], snap[0]};
            ComputeSnap(gContext.mScale, scaleSnap);
        }

        // no 0 allowed
        for (int i = 0; i < 3; i++)
            gContext.mScale[i] = max(gContext.mScale[i], 0.001f);

        if (gContext.mScaleLast != gContext.mScale) {
            modified = true;
        }
        gContext.mScaleLast = gContext.mScale;

        // compute matrix & delta
        matrix_t deltaMatrixScale;
        deltaMatrixScale.Scale(gContext.mScale * gContext.mScaleValueOrigin);

        matrix_t res = deltaMatrixScale * gContext.mModelLocal;
        *(matrix_t *)matrix = res;

        if (deltaMatrix) {
            vec_t deltaScale = gContext.mScale * gContext.mScaleValueOrigin;

            vec_t originalScaleDivider;
            originalScaleDivider.x = 1 / gContext.mModelScaleOrigin.x;
            originalScaleDivider.y = 1 / gContext.mModelScaleOrigin.y;
            originalScaleDivider.z = 1 / gContext.mModelScaleOrigin.z;

            deltaScale = deltaScale * originalScaleDivider;

            deltaMatrixScale.Scale(deltaScale);
            memcpy(deltaMatrix, deltaMatrixScale.m16, sizeof(float) * 16);
        }

        if (!io.MouseDown[0]) {
            gContext.mbUsing = false;
            gContext.mScale.Set(1.f, 1.f, 1.f);
        }

        type = gContext.mCurrentOperation;
    }
    return modified;
}

static bool HandleRotation(float *matrix, float *deltaMatrix, OPERATION op, int &type, const float *snap) {
    if (!Intersects(op, ROTATE) || type != MT_NONE || !gContext.mbMouseOver) {
        return false;
    }
    ImGuiIO &io = ImGui::GetIO();
    bool applyRotationLocaly = gContext.mMode == LOCAL;
    bool modified = false;

    if (!gContext.mbUsing) {
        type = GetRotateType(op);

        if (type != MT_NONE) {
            ImGui::SetNextFrameWantCaptureMouse(true);
        }

        if (type == MT_ROTATE_SCREEN) {
            applyRotationLocaly = true;
        }

        if (CanActivate() && type != MT_NONE) {
            gContext.mbUsing = true;
            gContext.mEditingID = gContext.mActualID;
            gContext.mCurrentOperation = type;
            const vec_t rotatePlanNormal[] = {gContext.mModel.v.right, gContext.mModel.v.up, gContext.mModel.v.dir, -gContext.mCameraDir};
            // pickup plan
            if (applyRotationLocaly) {
                gContext.mTranslationPlan = BuildPlan(gContext.mModel.v.position, rotatePlanNormal[type - MT_ROTATE_X]);
            } else {
                gContext.mTranslationPlan = BuildPlan(gContext.mModelSource.v.position, directionUnary[type - MT_ROTATE_X]);
            }

            const float len = IntersectRayPlane(gContext.mRayOrigin, gContext.mRayVector, gContext.mTranslationPlan);
            vec_t localPos = gContext.mRayOrigin + gContext.mRayVector * len - gContext.mModel.v.position;
            gContext.mRotationVectorSource = Normalized(localPos);
            gContext.mRotationAngleOrigin = ComputeAngleOnPlan();
        }
    }

    // rotation
    if (gContext.mbUsing && (gContext.mActualID == -1 || gContext.mActualID == gContext.mEditingID) && IsRotateType(gContext.mCurrentOperation)) {
        ImGui::SetNextFrameWantCaptureMouse(true);
        gContext.mRotationAngle = ComputeAngleOnPlan();
        if (snap) {
            float snapInRadian = snap[0] * DEG2RAD;
            ComputeSnap(&gContext.mRotationAngle, snapInRadian);
        }
        vec_t rotationAxisLocalSpace;

        rotationAxisLocalSpace.TransformVector(makeVect(gContext.mTranslationPlan.x, gContext.mTranslationPlan.y, gContext.mTranslationPlan.z, 0.f), gContext.mModelInverse);
        rotationAxisLocalSpace.Normalize();

        matrix_t deltaRotation;
        deltaRotation.RotationAxis(rotationAxisLocalSpace, gContext.mRotationAngle - gContext.mRotationAngleOrigin);
        if (gContext.mRotationAngle != gContext.mRotationAngleOrigin) {
            modified = true;
        }
        gContext.mRotationAngleOrigin = gContext.mRotationAngle;

        matrix_t scaleOrigin;
        scaleOrigin.Scale(gContext.mModelScaleOrigin);

        if (applyRotationLocaly) {
            *(matrix_t *)matrix = scaleOrigin * deltaRotation * gContext.mModelLocal;
        } else {
            matrix_t res = gContext.mModelSource;
            res.v.position.Set(0.f);

            *(matrix_t *)matrix = res * deltaRotation;
            ((matrix_t *)matrix)->v.position = gContext.mModelSource.v.position;
        }

        if (deltaMatrix) {
            *(matrix_t *)deltaMatrix = gContext.mModelInverse * deltaRotation * gContext.mModel;
        }

        if (!io.MouseDown[0]) {
            gContext.mbUsing = false;
            gContext.mEditingID = -1;
        }
        type = gContext.mCurrentOperation;
    }
    return modified;
}

bool Manipulate(const float *view, const float *projection, OPERATION operation, MODE mode, float *matrix, float *deltaMatrix, const float *snap) {
    // Scale is always local or matrix will be skewed when applying world scale or oriented matrix
    ComputeContext(view, projection, matrix, (operation & SCALE) ? LOCAL : mode);

    // set delta to identity
    if (deltaMatrix) {
        ((matrix_t *)deltaMatrix)->SetToIdentity();
    }

    // behind camera
    vec_t camSpacePosition;
    camSpacePosition.TransformPoint(makeVect(0.f, 0.f, 0.f), gContext.mMVP);
    if (!gContext.mIsOrthographic && camSpacePosition.z < 0.001f && !gContext.mbUsing) {
        return false;
    }

    // --
    int type = MT_NONE;
    bool manipulated = HandleTranslation(matrix, deltaMatrix, operation, type, snap) ||
        HandleScale(matrix, deltaMatrix, operation, type, snap) ||
        HandleRotation(matrix, deltaMatrix, operation, type, snap);

    gContext.mOperation = operation;
    DrawRotationGizmo(operation, type);
    DrawTranslationGizmo(operation, type);
    DrawScaleGizmo(operation, type);
    DrawScaleUniveralGizmo(operation, type);
    return manipulated;
}
} // namespace ImGuizmo
