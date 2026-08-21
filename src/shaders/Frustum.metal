#ifndef FRUSTUM_MSL
#define FRUSTUM_MSL

#include "MslPrelude.metal"

inline float4 vp_row(float4x4 m, uint i) {
    return float4(m[0][i], m[1][i], m[2][i], m[3][i]);
}

inline bool in_frustum(float4x4 view_proj, float3 center, float3 ax, float3 ay, float3 az) {
    const float4 r3 = vp_row(view_proj, 3);
    const float4 planes[6] = {
        r3 + vp_row(view_proj, 0), r3 - vp_row(view_proj, 0),
        r3 + vp_row(view_proj, 1), r3 - vp_row(view_proj, 1),
        vp_row(view_proj, 2), r3 - vp_row(view_proj, 2),
    };
    for (uint p = 0; p < 6; ++p) {
        const float3 n = planes[p].xyz;
        const float radius = abs(dot(n, ax)) + abs(dot(n, ay)) + abs(dot(n, az));
        if (dot(n, center) + planes[p].w < -radius) return false;
    }
    return true;
}

inline bool sphere_in_frustum(float4x4 view_proj, float3 center, float radius) {
    const float4 r3 = vp_row(view_proj, 3);
    const float4 planes[6] = {
        r3 + vp_row(view_proj, 0), r3 - vp_row(view_proj, 0),
        r3 + vp_row(view_proj, 1), r3 - vp_row(view_proj, 1),
        vp_row(view_proj, 2), r3 - vp_row(view_proj, 2),
    };
    for (uint p = 0; p < 6; ++p) {
        const float3 n = planes[p].xyz;
        if (dot(n, center) + planes[p].w < -radius * length(n)) return false;
    }
    return true;
}

#endif
