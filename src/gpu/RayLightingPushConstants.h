#pragma once
#include "gpu/VisibilityShadingPushConstants.h"
struct RayLightingPushConstants {
    VisibilityShadingPushConstants Visibility DEFAULT();
    mat4 ClipToWorld DEFAULT();
    uint32_t SampleOffset DEFAULT();
    uint32_t SingleSample DEFAULT();
    mat4 PreviousViewProj DEFAULT();
    uint32_t HistoryValid DEFAULT();
    uint32_t ApplyMotion DEFAULT();
    mat4 PreviousClipToWorld DEFAULT(1);
    uint32_t StagedSample DEFAULT();
    uint32_t OpaqueLayerActive DEFAULT();
    uint32_t QuincunxGlass DEFAULT();
    uint32_t DiffuseTileOffset DEFAULT();
};
