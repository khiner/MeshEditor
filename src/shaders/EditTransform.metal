#ifndef EDIT_TRANSFORM_MSL
#define EDIT_TRANSFORM_MSL

#include "TRSUtils.metal"

inline float3 apply_edit_transform(float3 world_pos, float3 pivot, Transform delta) {
    float3 offset = world_pos - pivot;
    offset = float3(delta.S) * offset;
    offset = quat_rotate(float4(delta.R), offset);
    return pivot + offset + float3(delta.P);
}

// Canonical, committed and preview positions use exactly the same frame math.
inline float3 apply_geometry_edit_transform(float3 local, Transform frame, float3 pivot, Transform delta) {
    return trs_inverse_transform_point(frame, apply_edit_transform(trs_transform_point(frame, local), pivot, delta));
}

#endif
