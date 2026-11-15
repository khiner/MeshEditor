#pragma once

#include "gpu/AttributeNode.h"

#ifndef __METAL_VERSION__
#include <algorithm>
#include <cfloat>
#include <cmath>
#endif

inline float SmoothMinimum(float a, float b, float c) {
#ifndef __METAL_VERSION__
    using namespace std;
#endif
    if (c == 0.f) return min(a, b);
    const float h = max(c - abs(a - b), 0.f) / c;
    return min(a, b) - h * h * h * c / 6.f;
}
// The operation of Blender's Math node on a, b and c.
// `clamped` limits the result to [0, 1].
inline float EvaluateFloatMath(FloatMathOp op, float a, float b, float c, bool clamped) {
#ifndef __METAL_VERSION__
    using namespace std;
#endif
    const float pi = 3.14159265358979f;
    float value = a;
    switch (op) {
        case FloatMathOp::Add: value = a + b; break;
        case FloatMathOp::Subtract: value = a - b; break;
        case FloatMathOp::Multiply: value = a * b; break;
        case FloatMathOp::Divide: value = b != 0.f ? a / b : 0.f; break;
        case FloatMathOp::MultiplyAdd: value = a * b + c; break;
        case FloatMathOp::Power: value = a >= 0.f ? pow(a, b) : b == floor(b) ? pow(-a, b) * (fmod(b, 2.f) != 0.f ? -1.f : 1.f) :
                                                                                0.f;
            break;
        case FloatMathOp::Logarithm: value = a > 0.f && b > 0.f && log(b) != 0.f ? log(a) / log(b) : 0.f; break;
        case FloatMathOp::SquareRoot: value = sqrt(max(a, 0.f)); break;
        case FloatMathOp::InverseSquareRoot: value = a > 0.f ? 1.f / sqrt(a) : 0.f; break;
        case FloatMathOp::Absolute: value = abs(a); break;
        case FloatMathOp::Exponent: value = exp(a); break;
        case FloatMathOp::Minimum: value = min(a, b); break;
        case FloatMathOp::Maximum: value = max(a, b); break;
        case FloatMathOp::LessThan: value = float(a < b); break;
        case FloatMathOp::GreaterThan: value = float(a > b); break;
        case FloatMathOp::Sign: value = a > 0.f ? 1.f : a < 0.f ? -1.f :
                                                                  0.f;
            break;
        case FloatMathOp::Compare: value = float(a == b || abs(a - b) <= max(c, FLT_EPSILON)); break;
        case FloatMathOp::SmoothMinimum: value = SmoothMinimum(a, b, c); break;
        case FloatMathOp::SmoothMaximum: value = -SmoothMinimum(-a, -b, c); break;
        case FloatMathOp::Round: value = floor(a + .5f); break;
        case FloatMathOp::Floor: value = floor(a); break;
        case FloatMathOp::Ceil: value = ceil(a); break;
        case FloatMathOp::Truncate: value = trunc(a); break;
        case FloatMathOp::Fraction: value = a - floor(a); break;
        case FloatMathOp::TruncatedModulo: value = b != 0.f ? fmod(a, b) : 0.f; break;
        case FloatMathOp::FlooredModulo: value = b != 0.f ? a - floor(a / b) * b : 0.f; break;
        case FloatMathOp::Wrap: value = b != c ? a - (b - c) * floor((a - c) / (b - c)) : c; break;
        case FloatMathOp::Snap: value = b != 0.f ? floor(a / b) * b : 0.f; break;
        case FloatMathOp::PingPong: {
            const float cycle = b != 0.f ? (a - b) / (b * 2.f) : 0.f;
            value = b != 0.f ? abs((cycle - floor(cycle)) * b * 2.f - b) : 0.f;
            break;
        }
        case FloatMathOp::Sine: value = sin(a); break;
        case FloatMathOp::Cosine: value = cos(a); break;
        case FloatMathOp::Tangent: value = tan(a); break;
        case FloatMathOp::Arcsine: value = asin(clamp(a, -1.f, 1.f)); break;
        case FloatMathOp::Arccosine: value = acos(clamp(a, -1.f, 1.f)); break;
        case FloatMathOp::Arctangent: value = atan(a); break;
        case FloatMathOp::Arctan2: value = atan2(a, b); break;
        case FloatMathOp::HyperbolicSine: value = sinh(a); break;
        case FloatMathOp::HyperbolicCosine: value = cosh(a); break;
        case FloatMathOp::HyperbolicTangent: value = tanh(a); break;
        case FloatMathOp::ToRadians: value = a * (pi / 180.f); break;
        case FloatMathOp::ToDegrees: value = a * (180.f / pi); break;
    }
    return clamped ? clamp(value, 0.f, 1.f) : value;
}
