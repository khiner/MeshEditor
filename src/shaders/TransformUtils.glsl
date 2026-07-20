// Shared pending-transform helpers.

vec4 quat_conjugate(vec4 q) { return vec4(-q.xyz, q.w); }

vec3 apply_pending_transform_world(vec3 world_pos) {
    vec3 offset = world_pos - SceneViewUBO.PendingPivot;
    offset = SceneViewUBO.PendingScale * offset;
    offset = quat_rotate(SceneViewUBO.PendingRotation, offset);
    return SceneViewUBO.PendingPivot + offset + SceneViewUBO.PendingTranslation;
}

vec3 trs_inverse_transform_point(Transform t, vec3 pos) {
    return quat_rotate(quat_conjugate(t.R), pos - t.P) / t.S;
}

// Object-mode gizmo preview: selected instances' world positions follow the pending transform.
// The pose pre-pass bakes edit-mode vertex previews into posed positions.
vec3 apply_object_pending_transform(DrawData draw, vec3 world_pos) {
    if (SceneViewUBO.IsTransforming == 0u || SceneViewUBO.InteractionMode == InteractionMode_Edit || draw.InstanceStateSlot == INVALID_SLOT) return world_pos;
    const uint instance_state = uint(InstanceStateBuffers[draw.InstanceStateSlot].States[draw.FirstInstance]);
    if ((instance_state & STATE_SELECTED) == 0u) return world_pos;
    return apply_pending_transform_world(world_pos);
}
