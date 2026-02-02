// Shared pending-transform helpers.

// Apply rotation quaternion (xyzw) to a vector.
vec3 rotate_by_quat(vec3 v, vec4 q) {
    const vec3 u = q.xyz;
    const float s = q.w;
    return 2.0 * dot(u, v) * u + (s * s - dot(u, u)) * v + 2.0 * s * cross(u, v);
}

// Apply pending transform (P, R, S) around a world-space pivot.
vec3 apply_pending_transform(vec3 world_pos, vec3 pivot) {
    return pivot + rotate_by_quat(SceneViewUBO.PendingTransformS * (world_pos - pivot), SceneViewUBO.PendingTransformR) + SceneViewUBO.PendingTransformP; // Translate
}

bool should_apply_pending_transform(uint instance_state, uint vertex_state, bool is_edit_mode) {
    const bool instance_selected = (instance_state & STATE_SELECTED) != 0u;
    const bool vertex_selected = (vertex_state & STATE_SELECTED) != 0u;
    return SceneViewUBO.IsTransforming != 0u && ((!is_edit_mode && instance_selected) || (is_edit_mode && vertex_selected));
}
