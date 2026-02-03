// Shared pending-transform helpers.

vec3 apply_pending_transform(vec3 world_pos) {
    return (SceneViewUBO.PendingTransform * vec4(world_pos, 1.0)).xyz;
}

bool should_apply_pending_transform(uint instance_state, uint vertex_state, bool is_edit_mode) {
    const bool instance_selected = (instance_state & STATE_SELECTED) != 0u;
    const bool vertex_selected = (vertex_state & STATE_SELECTED) != 0u;
    return SceneViewUBO.IsTransforming != 0u && ((!is_edit_mode && instance_selected) || (is_edit_mode && vertex_selected));
}
