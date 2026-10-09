#ifndef ATTRIBUTE_NODES_MSL
#define ATTRIBUTE_NODES_MSL

// Each attribute node's value at one element from that element's input values and the node's parameters.
// Values travel as the words of their components, and a bool is the low bit of its first word.
#include "gpu/AttributeNode.h"

inline float3 VectorOf(uint4 value) { return as_type<float3>(value.xyz); }
inline uint4 WordsOf(float3 value) { return uint4(as_type<uint3>(value), 0u); }
inline uint4 WordsOf(float value) { return uint4(as_type<uint>(value), 0u, 0u, 0u); }
inline uint4 WordsOf(bool value) { return uint4(uint(value), 0u, 0u, 0u); }

// Scale multiplies A by the float B, and Dot, Length and Distance return a float.
inline uint4 VectorMathNode(uint4 a_words, uint4 b_words, VectorMathOp op) {
    const float3 a = VectorOf(a_words), b = VectorOf(b_words);
    switch (op) {
        case VectorMathOp::Add: return WordsOf(a + b);
        case VectorMathOp::Subtract: return WordsOf(a - b);
        case VectorMathOp::Multiply: return WordsOf(a * b);
        case VectorMathOp::Scale: return WordsOf(a * as_type<float>(b_words.x));
        case VectorMathOp::Dot: return WordsOf(dot(a, b));
        case VectorMathOp::Cross: return WordsOf(cross(a, b));
        case VectorMathOp::Length: return WordsOf(length(a));
        case VectorMathOp::Normalize: return WordsOf(length(a) > 0.f ? normalize(a) : float3(0.f));
        case VectorMathOp::Distance: return WordsOf(distance(a, b));
    }
    return a_words;
}

// Abs, Sqrt, Sin and Cos read only A.
// A zero divisor gives zero, and a negative base takes only an integer power.
inline uint4 FloatMathNode(uint4 a_words, uint4 b_words, FloatMathOp op) {
    const float a = as_type<float>(a_words.x), b = as_type<float>(b_words.x);
    switch (op) {
        case FloatMathOp::Add: return WordsOf(a + b);
        case FloatMathOp::Subtract: return WordsOf(a - b);
        case FloatMathOp::Multiply: return WordsOf(a * b);
        case FloatMathOp::Divide: return WordsOf(b != 0.f ? a / b : 0.f);
        case FloatMathOp::Min: return WordsOf(min(a, b));
        case FloatMathOp::Max: return WordsOf(max(a, b));
        case FloatMathOp::Abs: return WordsOf(abs(a));
        case FloatMathOp::Power: return WordsOf(a >= 0.f ? powr(a, b) : b == floor(b) ? powr(-a, b) * (fmod(b, 2.f) != 0.f ? -1.f : 1.f) : 0.f);
        case FloatMathOp::Sqrt: return WordsOf(sqrt(max(a, 0.f)));
        case FloatMathOp::Sin: return WordsOf(sin(a));
        case FloatMathOp::Cos: return WordsOf(cos(a));
    }
    return a_words;
}

// Compares A and B as floats, each a float value or the chosen component of a vector.
inline uint4 CompareNode(uint4 a_words, AttributeType a_type, uint4 b_words, AttributeType b_type, uint component, CompareOp op, float epsilon) {
    const float a = as_type<float>(a_type == AttributeType::Float ? a_words.x : a_words[component]);
    const float b = as_type<float>(b_type == AttributeType::Float ? b_words.x : b_words[component]);
    switch (op) {
        case CompareOp::Less: return WordsOf(a < b);
        case CompareOp::Greater: return WordsOf(a > b);
        case CompareOp::Equal: return WordsOf(abs(a - b) <= epsilon);
    }
    return WordsOf(false);
}

// Not reads only A.
inline uint4 BooleanMathNode(uint4 a_words, uint4 b_words, BooleanMathOp op) {
    const bool a = a_words.x & 1u, b = b_words.x & 1u;
    switch (op) {
        case BooleanMathOp::And: return WordsOf(a && b);
        case BooleanMathOp::Or: return WordsOf(a || b);
        case BooleanMathOp::Not: return WordsOf(!a);
        case BooleanMathOp::Xor: return WordsOf(a != b);
    }
    return WordsOf(false);
}

#endif
