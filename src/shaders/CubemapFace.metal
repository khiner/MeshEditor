#pragma once
#include <metal_stdlib>
using namespace metal;

// Y-up cubemap face convention, with layers ordered +X, -X, +Y, -Y, +Z, -Z.
inline float3 FaceDirection(uint face, float2 uv) {
    switch (face) {
        case 0: return normalize(float3(1.0f, -uv.y, -uv.x));
        case 1: return normalize(float3(-1.0f, -uv.y, uv.x));
        case 2: return normalize(float3(uv.x, 1.0f, uv.y));
        case 3: return normalize(float3(uv.x, -1.0f, -uv.y));
        case 4: return normalize(float3(uv.x, -uv.y, 1.0f));
        default: return normalize(float3(-uv.x, -uv.y, -1.0f));
    }
}
