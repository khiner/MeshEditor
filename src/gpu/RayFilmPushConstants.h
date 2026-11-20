#pragma once
#include "gpu/Types.h"
#include "gpu/VisibilityShadingPushConstants.h"
struct RayFilmPushConstants {
    float PreviousWeight DEFAULT();
    float Weight DEFAULT();
    GpuArray<float, 9> Weights DEFAULT();
};

struct RayFilmReprojectPushConstants {
    RayFilmPushConstants Filter;
    mat4 CurrentViewProj DEFAULT(1);
    mat4 PreviousViewProj DEFAULT(1);
    mat4 PreviousGeometryViewProj DEFAULT(1);
    mat4 CurrentGeometryClipToWorld DEFAULT(1);
    mat4 PreviousGeometryClipToWorld DEFAULT(1);
    uint32_t ApplyMotion DEFAULT();
    VisibilityShadingPushConstants Visibility;
    uint32_t Deformation DEFAULT();
    uint32_t TransparentSurfaces DEFAULT();
};
