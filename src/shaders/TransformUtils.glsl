// Shared pending-transform helpers.

vec3 apply_pending_transform(vec3 world_pos) {
    return (SceneViewUBO.PendingTransform * vec4(world_pos, 1.0)).xyz;
}

bool should_apply_pending_transform(uint instance_state, DrawData draw, uint idx) {
    if (SceneViewUBO.IsTransforming == 0u) return false;

    const bool is_edit_mode = SceneViewUBO.InteractionMode == InteractionModeEdit;
    if (!is_edit_mode) return (instance_state & STATE_SELECTED) != 0u;

    if (pc.TransformVertexStateSlot == INVALID_SLOT) return false;
    const uint vertex_state = uint(ElementStateBuffers[pc.TransformVertexStateSlot].States[draw.VertexOffset + idx]);
    return (vertex_state & STATE_SELECTED) != 0u;
}
