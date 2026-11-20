#pragma once
#include "gpu/Types.h"
#include "gpu/VisibilityShadingPushConstants.h"
struct RayReconstructPushConstants {
    mat4 ClipToWorld DEFAULT();
    uint32_t Denoise DEFAULT();
    uint32_t Step DEFAULT(1u);
    uint32_t Frame DEFAULT();
    uint32_t HistoryValid DEFAULT();
    uint32_t Moving DEFAULT();
    mat4 PreviousViewProj DEFAULT();
    VisibilityShadingPushConstants Visibility DEFAULT();
    uint32_t Lobe DEFAULT();
    uint32_t ApplyMotion DEFAULT();
    mat4 PreviousClipToWorld DEFAULT(1);
    uint32_t FastGI DEFAULT();
    uint32_t Reflections DEFAULT();
    float MaxRoughness DEFAULT(.5f);
    uint32_t QuincunxGlass DEFAULT();
    uint32_t SplitTiles DEFAULT();
};
