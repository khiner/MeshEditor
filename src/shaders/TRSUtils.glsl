// Quaternion and TRS transform utilities.
// Requires: WorldTransform.glsl included.

vec3 quat_rotate(vec4 q, vec3 v) {
    vec3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

mat4 trs_to_mat4(WorldTransform t) {
    // Build rotation matrix from quaternion, then apply scale and translation.
    vec4 q = t.Rotation;
    float x2 = 2.0 * q.x * q.x, y2 = 2.0 * q.y * q.y, z2 = 2.0 * q.z * q.z;
    float xy = 2.0 * q.x * q.y, xz = 2.0 * q.x * q.z, yz = 2.0 * q.y * q.z;
    float wx = 2.0 * q.w * q.x, wy = 2.0 * q.w * q.y, wz = 2.0 * q.w * q.z;

    mat3 R = mat3(
        vec3(1.0 - y2 - z2, xy + wz,       xz - wy),
        vec3(xy - wz,       1.0 - x2 - z2, yz + wx),
        vec3(xz + wy,       yz - wx,       1.0 - x2 - y2)
    );

    mat4 M;
    M[0] = vec4(R[0] * t.Scale.x, 0.0);
    M[1] = vec4(R[1] * t.Scale.y, 0.0);
    M[2] = vec4(R[2] * t.Scale.z, 0.0);
    M[3] = vec4(t.Position, 1.0);
    return M;
}

vec3 trs_transform_point(WorldTransform t, vec3 pos) {
    return t.Position + quat_rotate(t.Rotation, t.Scale * pos);
}

vec3 trs_transform_normal(WorldTransform t, vec3 normal) {
    return quat_rotate(t.Rotation, normal / t.Scale);
}
