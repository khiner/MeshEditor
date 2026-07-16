#version 450

#include "Bindless.glsl"
#include "MorphDeform.glsl"
#include "ArmatureDeform.glsl"
#include "TransformUtils.glsl"

// Screen motion across the shutter, as (prev->current, current->next) in UV space.
// Both halves point the same way, so the gather can walk either direction with one vector.
layout(location = 0) out vec4 Motion;

// World position of `vert` under one pose. The per-draw offsets are shared across poses,
// so a pose is selected purely by its buffer slots.
vec3 PoseWorldPos(DrawData draw, Vertex vert, uint idx, uint model_slot, uint armature_slot, uint morph_slot) {
    vec3 normal = vert.Normal;
    vec3 pos = vert.Position;
    ApplyMorphDeform(draw, pos, idx, normal, morph_slot);
    const vec3 local_pos = ApplyArmatureDeform(draw, pos, idx, normal, armature_slot);
    const Transform world = ModelBuffers[nonuniformEXT(model_slot)].Models[draw.FirstInstance];
    return apply_pending_transform(draw, world, local_pos, idx);
}

// Each pose projects through its own view: looking through an animated camera moves the view
// across the shutter too, and a pure camera move is motion like any other.
vec2 ProjectToUv(mat4 view_proj, vec3 world_pos) {
    const vec4 clip = view_proj * vec4(world_pos, 1.0);
    return clip.xy / clip.w;
}

void main() {
    const DrawData draw = GetDrawData();
    const uint idx = IndexBuffers[draw.IndexSlotOffset.Slot].Indices[draw.IndexSlotOffset.Offset + uint(gl_VertexIndex)];
    const Vertex vert = VertexBuffers[draw.VertexSlot].Vertices[idx + draw.VertexOffset];

    const vec3 curr = PoseWorldPos(draw, vert, idx, draw.ModelSlot, SceneViewUBO.ArmatureDeformSlot, SceneViewUBO.MorphWeightsSlot);
    const vec3 prev = PoseWorldPos(draw, vert, idx, SceneViewUBO.PrevModelSlot, SceneViewUBO.PrevArmatureDeformSlot, SceneViewUBO.PrevMorphWeightsSlot);
    const vec3 next = PoseWorldPos(draw, vert, idx, SceneViewUBO.NextModelSlot, SceneViewUBO.NextArmatureDeformSlot, SceneViewUBO.NextMorphWeightsSlot);

    gl_Position = SceneViewUBO.ViewProj * vec4(curr, 1.0);

    const vec2 curr_uv = ProjectToUv(SceneViewUBO.ViewProj, curr);
    const vec2 prev_uv = ProjectToUv(SceneViewUBO.PrevViewProj, prev);
    const vec2 next_uv = ProjectToUv(SceneViewUBO.NextViewProj, next);
    // NDC spans 2 units across the viewport, so halving converts these to UV. The second half is
    // stored pointing backward like the first, which the gather's motion scale undoes.
    Motion = vec4(prev_uv - curr_uv, curr_uv - next_uv) * 0.5;
}
