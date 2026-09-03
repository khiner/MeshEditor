#ifndef TEXQUAD_MSL
#define TEXQUAD_MSL

#include "Varyings.metal"

constant float2 QuadPositions[4] = {float2(-1, -1), float2(1, -1), float2(-1, 1), float2(1, 1)};

vertex QuadVaryings TexQuadVertex(uint vertex_id [[vertex_id]]) {
    const float2 p = QuadPositions[vertex_id];
    return QuadVaryings{float4(p, 0.0f, 1.0f), ndc_to_uv(p)};
}

#endif
