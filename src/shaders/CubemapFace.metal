#ifndef CUBEMAPFACE_MSL
#define CUBEMAPFACE_MSL

#include <metal_stdlib>
using namespace metal;

// Y-up cubemap face convention (layer order: +X, -X, +Y, -Y, +Z, -Z).
// uv is in [-1, 1] for the face's s/t axes.
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

#endif
