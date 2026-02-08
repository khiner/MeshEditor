// Shared pending-transform helpers.

vec3 apply_pending_transform(vec3 local_pos, vec3 world_pos, WorldMatrix world, DrawData draw) {
    // Edit preview is mesh-local with a precomputed per-mesh local pending matrix.
    if (pc.TransformVertexStateSlot != INVALID_SLOT) {
        const mat4 pending_local = ModelBuffers[draw.PendingLocalTransformSlot].Models[draw.PendingLocalTransformOffset].M;
        return vec3(world.M * (pending_local * vec4(local_pos, 1.0)));
    }

    // Object preview is instance-local: transform current world position directly.
    return (SceneViewUBO.PendingTransform * vec4(world_pos, 1.0)).xyz;
}

bool should_apply_pending_transform(DrawData draw, uint idx) {
    if (SceneViewUBO.IsTransforming == 0u) return false;

    // In edit preview, pending transforms are mesh-scoped and keyed off mesh vertex state.
    // In object preview, pending transforms are instance-scoped and keyed off instance selection.
    if (pc.TransformVertexStateSlot != INVALID_SLOT) {
        if (draw.PendingLocalTransformSlot == INVALID_SLOT) return false;
        const uint vertex_state = uint(ElementStateBuffers[pc.TransformVertexStateSlot].States[draw.VertexOffset + idx]);
        return (vertex_state & STATE_SELECTED) != 0u;
    }
    if (SceneViewUBO.InteractionMode == InteractionModeEdit || draw.InstanceStateSlot == INVALID_SLOT) return false;
    const uint instance_state = uint(InstanceStateBuffers[draw.InstanceStateSlot].States[draw.InstanceStateOffset + draw.FirstInstance]);
    return (instance_state & STATE_SELECTED) != 0u;
}
