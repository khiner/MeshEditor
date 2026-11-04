#pragma once
// Float bits whose unsigned order is the floats' order.
inline uint OrderedBits(float value) {
    const uint bits = as_type<uint>(value);
    return (bits & 0x80000000u) ? ~bits : bits | 0x80000000u;
}
inline float OrderedFloat(uint bits) { return as_type<float>((bits & 0x80000000u) ? bits & 0x7fffffffu : ~bits); }
