#pragma once
#include "Bindless.metal"
#include "CubemapFace.metal"
#include "gpu/RayEnvironment.h"

// Piecewise-constant cube-cell proposal. The PDF below includes the cube projection Jacobian.
kernel void RayEnvironmentRowsKernel(device const BindlessSet &b [[buffer(0)]], constant uint &slot [[buffer(1)]],
                                     device float *cdf [[buffer(8)]], uint row [[thread_position_in_grid]]) {
    const uint face = row / RayEnvironmentSize, y = row % RayEnvironmentSize;
    float total = 0.0f;
    cdf[row * RayEnvironmentStride] = 0.0f;
    for (uint x = 0; x < RayEnvironmentSize; ++x) {
        const float2 uv = 2.0f * (float2(x, y) + 0.5f) / float(RayEnvironmentSize) - 1.0f;
        const float3 direction = FaceDirection(face, uv);
        const float3 radiance = b.CubeSampler[slot].Texture.sample(b.CubeSampler[slot].Sampler, direction, level(0)).rgb;
        const float luminance = max(dot(radiance, float3(0.2126f, 0.7152f, 0.0722f)), 0.0f);
        total += (luminance + 1e-6f) * pow(1.0f + dot(uv, uv), -1.5f);
        cdf[row * RayEnvironmentStride + x + 1u] = total;
    }
}
kernel void RayEnvironmentMarginalKernel(device float *cdf [[buffer(8)]]) {
    float total = 0.0f;
    cdf[RayEnvironmentMarginal] = 0.0f;
    for (uint row = 0; row < RayEnvironmentRows; ++row) {
        total += cdf[row * RayEnvironmentStride + RayEnvironmentSize];
        cdf[RayEnvironmentMarginal + row + 1u] = total;
    }
}
