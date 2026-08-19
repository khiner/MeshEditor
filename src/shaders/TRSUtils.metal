#ifndef TRSUTILS_MSL
#define TRSUTILS_MSL

// Quaternion and TRS transform utilities.
#include "Transform.metal"

inline float3 quat_rotate(float4 q, float3 v) {
    const float3 t = 2.0f * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

inline float4x4 trs_to_mat4(Transform t) {
    // Build rotation matrix from quaternion, then apply scale and translation.
    const float4 q = float4(t.R);
    const float x2 = 2.0f * q.x * q.x, y2 = 2.0f * q.y * q.y, z2 = 2.0f * q.z * q.z;
    const float xy = 2.0f * q.x * q.y, xz = 2.0f * q.x * q.z, yz = 2.0f * q.y * q.z;
    const float wx = 2.0f * q.w * q.x, wy = 2.0f * q.w * q.y, wz = 2.0f * q.w * q.z;

    const float3x3 R = float3x3(
        float3(1.0f - y2 - z2, xy + wz, xz - wy),
        float3(xy - wz, 1.0f - x2 - z2, yz + wx),
        float3(xz + wy, yz - wx, 1.0f - x2 - y2)
    );

    const float3 s = float3(t.S);
    return float4x4(
        float4(R[0] * s.x, 0.0f),
        float4(R[1] * s.y, 0.0f),
        float4(R[2] * s.z, 0.0f),
        float4(float3(t.P), 1.0f)
    );
}

inline float3 trs_transform_point(Transform t, float3 pos) {
    return float3(t.P) + quat_rotate(float4(t.R), float3(t.S) * pos);
}

inline float3 trs_transform_normal(Transform t, float3 normal) {
    return quat_rotate(float4(t.R), normal / float3(t.S));
}

#endif
