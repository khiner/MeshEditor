#pragma once
#include "gpu/Types.h"
struct RayTracingInstance {
    uint32_t InstanceSlot DEFAULT();
    uint32_t TriangleOffset DEFAULT();
    uint32_t VolumeMaterial DEFAULT(InvalidOffset);
};

GPU_CONSTANT uint32_t RayMediumCapacity = 12u;
struct RayMediumState {
    uint32_t Count DEFAULT();
    GpuArray<uint32_t, RayMediumCapacity> Materials DEFAULT();
    GpuArray<uint32_t, RayMediumCapacity> Objects DEFAULT();
};

GPU_CONSTANT uint32_t RayMotionRigid = 1u;
GPU_CONSTANT uint32_t RayMotionDeformed = 2u;

struct RayMotionTransform {
    mat4 CurrentToPrevious DEFAULT(1);
    mat3 NormalToPrevious DEFAULT(1);
    uint32_t Valid DEFAULT();
    uint32_t Identity DEFAULT();
};

// Previous primary surfaces share the ray queue after tracing has consumed it.
struct RayDeformedSurface {
    vec4 PositionValid DEFAULT();
    vec4 Normal DEFAULT();
};
static_assert(sizeof(RayDeformedSurface) == 32);

// Transient primary-ray data shared by generation and secondary lighting.
struct RayLightingSample {
    vec4 DirectionPdf DEFAULT();
    vec4 OriginFootprint DEFAULT();
    vec4 PositionFlags DEFAULT();
};
static_assert(sizeof(RayLightingSample) == 48);
// The flag bits of PositionFlags.w, whose upper 16 bits hold the sample's light scale as a half.
GPU_CONSTANT uint32_t RaySampleActive = 1u;
GPU_CONSTANT uint32_t RaySampleValid = 2u;
GPU_CONSTANT uint32_t RaySampleGlass = 4u;
static_assert(sizeof(RayDeformedSurface) <= sizeof(RayLightingSample));

inline uint32_t RayLightingLobeCount(bool transmission, bool diffuse) { return transmission || diffuse ? 2u : 1u; }

// Work contains ray dispatch arguments, compact pixels, then 8x8 denoising tiles.
// Opaque traced diffuse appends its own indirect dispatch and tile list.
GPU_CONSTANT uint32_t RayPixelOffset = 8u;
inline uint32_t RayTileCount(uint32_t width, uint32_t height) { return ((width + 7u) / 8u) * ((height + 7u) / 8u); }
inline uint32_t RayTileOffset(uint32_t width, uint32_t height) { return RayPixelOffset + width * height; }
inline uint32_t RayTileEnd(uint32_t width, uint32_t height) { return RayTileOffset(width, height) + RayTileCount(width, height); }
