#pragma once
#include "CubemapFace.metal"
#include "gpu/RayEnvironment.h"

inline uint RayCdfBin(device const float *cdf, uint count, float value) {
    uint lo = 0u, hi = count;
    while (lo + 1u < hi) { const uint mid = (lo + hi) / 2u; if (cdf[mid] <= value) lo = mid; else hi = mid; }
    return lo;
}
inline float3 SampleRayEnvironment(device const float *cdf, float2 u) {
    device const float *rows = cdf + RayEnvironmentMarginal;
    const float row_value = u.y * rows[RayEnvironmentRows];
    const uint row = RayCdfBin(rows, RayEnvironmentRows, row_value);
    device const float *columns = cdf + row * RayEnvironmentStride;
    const float column_value = u.x * columns[RayEnvironmentSize];
    const uint x = RayCdfBin(columns, RayEnvironmentSize, column_value);
    const float2 within = float2((column_value - columns[x]) / max(columns[x+1u] - columns[x], 1e-20f),
                                (row_value - rows[row]) / max(rows[row+1u] - rows[row], 1e-20f));
    const float2 uv = 2.0f * (float2(x, row % RayEnvironmentSize) + within) / float(RayEnvironmentSize) - 1.0f;
    const uint face = row / RayEnvironmentSize;
    return FaceDirection(face, uv);
}
inline float RayEnvironmentPdf(device const float *cdf, float3 d) {
    const float3 a = abs(d);
    uint face;
    float2 uv;
    if (a.x >= a.y && a.x >= a.z) { face = d.x > 0.0f ? 0u : 1u; uv = float2(d.x > 0.0f ? -d.z : d.z, -d.y) / a.x; }
    else if (a.y >= a.z) { face = d.y > 0.0f ? 2u : 3u; uv = float2(d.x, d.y > 0.0f ? d.z : -d.z) / a.y; }
    else { face = d.z > 0.0f ? 4u : 5u; uv = float2(d.z > 0.0f ? d.x : -d.x, -d.y) / a.z; }
    const uint2 pixel = min(uint2((uv * 0.5f + 0.5f) * float(RayEnvironmentSize)), uint2(RayEnvironmentSize - 1u));
    const uint row = face * RayEnvironmentSize + pixel.y;
    device const float *columns = cdf + row * RayEnvironmentStride;
    const float probability = (columns[pixel.x+1u] - columns[pixel.x]) / cdf[RayEnvironmentMarginal + RayEnvironmentRows];
    return probability * float(RayEnvironmentSize * RayEnvironmentSize) * 0.25f * pow(1.0f + dot(uv, uv), 1.5f);
}
