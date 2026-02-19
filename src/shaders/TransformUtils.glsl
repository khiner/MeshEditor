// Shared pending-transform helpers.

vec3 apply_pending_transform(vec3 world_pos) {
    vec3 offset = world_pos - SceneViewUBO.PendingPivot;
    offset = SceneViewUBO.PendingScale * offset;
    offset = quat_rotate(SceneViewUBO.PendingRotation, offset);
    return SceneViewUBO.PendingPivot + offset + SceneViewUBO.PendingTranslation;
}

bool should_apply_pending_transform(DrawData draw, uint idx) {
    if (SceneViewUBO.IsTransforming == 0u) return false;

    // In edit preview, pending transforms are mesh-scoped and keyed off mesh vertex state.
    // In object preview, pending transforms are instance-scoped and keyed off instance selection.
    if (pc.TransformVertexStateSlot != INVALID_SLOT) {
        if (draw.HasPendingVertexTransform == 0u) return false;
        const uint vertex_state = uint(ElementStateBuffers[pc.TransformVertexStateSlot].States[draw.VertexOffset + idx]);
        return (vertex_state & STATE_SELECTED) != 0u;
    }
    if (SceneViewUBO.InteractionMode == InteractionModeEdit || draw.InstanceStateSlot == INVALID_SLOT) return false;
    const uint instance_state = uint(InstanceStateBuffers[draw.InstanceStateSlot].States[draw.FirstInstance]);
    return (instance_state & STATE_SELECTED) != 0u;
}
