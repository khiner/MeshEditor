#pragma once
#include "gpu/Types.h"
struct FastGiPushConstants {
    mat4 ClipToWorld DEFAULT(1);
    mat4 PreviousViewProj DEFAULT(1);
    mat4 PreviousClipToWorld DEFAULT(1);
    mat4 Projection DEFAULT(1);
    mat4 InverseProjection DEFAULT(1);
    mat4 WorldToView DEFAULT(1);
    uint32_t Step DEFAULT(2);
    uint32_t Frame DEFAULT();
    uint32_t HistoryValid DEFAULT();
    uint32_t ApplyMotion DEFAULT();
    float PixelRadius DEFAULT();
};
