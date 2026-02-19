// Shared pending-transform helpers.

vec4 quat_conjugate(vec4 q) { return vec4(-q.xyz, q.w); }

vec3 apply_pending_transform_world(vec3 world_pos) {
    vec3 offset = world_pos - SceneViewUBO.PendingPivot;
    offset = SceneViewUBO.PendingScale * offset;
    offset = quat_rotate(SceneViewUBO.PendingRotation, offset);
    return SceneViewUBO.PendingPivot + offset + SceneViewUBO.PendingTranslation;
}

vec3 trs_inverse_transform_point(WorldTransform t, vec3 pos) {
    return quat_rotate(quat_conjugate(t.Rotation), pos - t.Position) / t.Scale;
}

bool should_apply_pending_transform(DrawData draw, uint idx) {
    if (SceneViewUBO.IsTransforming == 0u) return false;

    // In edit preview, pending transforms are keyed off mesh vertex state and driven by the mesh's primary edit instance.
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

vec3 apply_pending_transform(DrawData draw, WorldTransform world, vec3 local_pos, uint idx) {
    vec3 world_pos = trs_transform_point(world, local_pos);
    if (!should_apply_pending_transform(draw, idx)) return world_pos;
    if (pc.TransformVertexStateSlot != INVALID_SLOT) {
        const WorldTransform primary = ModelBuffers[draw.ModelSlot].Models[draw.PrimaryEditInstanceIndex];
        const vec3 primary_world = trs_transform_point(primary, local_pos);
        const vec3 pending_world = apply_pending_transform_world(primary_world);
        const vec3 pending_local = trs_inverse_transform_point(primary, pending_world);
        return trs_transform_point(world, pending_local);
    }
    return apply_pending_transform_world(world_pos);
}
