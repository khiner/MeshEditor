#pragma once
#include "TRSUtils.metal"

inline float3 apply_edit_transform(float3 world_pos, float3 pivot, Transform delta) {
    float3 offset = world_pos - pivot;
    offset = float3(delta.S) * offset;
    offset = quat_rotate(float4(delta.R), offset);
    return pivot + offset + float3(delta.P);
}

// The unit `mirror` reflects across the plane through the pivot with that normal before the delta, and a zero one reflects nothing.
inline float3 apply_geometry_edit_transform(float3 local, Transform frame, float3 pivot, float3 mirror, Transform delta) {
    const float3 p = trs_transform_point(frame, local);
    return trs_inverse_transform_point(frame, apply_edit_transform(p - 2.f * dot(p - pivot, mirror) * mirror, pivot, delta));
}
