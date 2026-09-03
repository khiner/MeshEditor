#ifndef POSEPREPASS_MSL
#define POSEPREPASS_MSL

// Writes current-pose mesh-local positions from morph, armature, and pending edit transforms.
#include "Bindless.metal"
#include "MorphDeform.metal"
#include "ArmatureDeform.metal"
#include "TransformUtils.metal"
#include "BoundsReducePushConstants.metal"
#include "EditSelection.metal"

kernel void PosePrepassKernel(
    uint local_id [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant BoundsReducePushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    const uint2 tile = uint2(scene.TileMap(pc.TileMapSlot)[group_id]);
    const DrawData draw = scene.Draws(pc.DrawDataSlot)[tile.x];
    const uint i = tile.y * 256u + local_id;
    if (i >= draw.VertexCountOrHeadImageSlot) return;

    float3 pos = float3(scene.Vertices(draw.VertexSlot)[draw.VertexOffset + i].Position);
    float3 normal = float3(0);
    float3 morph_normal_delta = float3(0);
    ApplyMorphDeform(scene, draw, pos, morph_normal_delta, i);
    pos = ApplyArmatureDeform(scene, draw, pos, i, normal);
    if (view.IsTransforming != 0u && draw.HasPendingVertexTransform != 0u &&
        EditSelectionBit(scene, draw.Selection.VertexBits, i)) {
        // Convert the primary edit instance's world-space gesture back into mesh-local posed positions.
        const Transform primary = scene.Models(draw.ModelSlot)[draw.PrimaryEditInstanceIndex];
        pos = trs_inverse_transform_point(primary, apply_pending_transform_world(scene, trs_transform_point(primary, pos)));
    }
    device packed_float3 *posed_positions = BindlessBufferMutable(packed_float3, bindless.Buffer, view.PosedPositionSlot);
    posed_positions[draw.PosedPositionOffset + i] = packed_float3(pos);
    // Precompute sum(w_t * NormalDelta_t) per vertex for authored morph normals.
    if (draw.MorphShadingAuthored != 0u) {
        device packed_float3 *deltas = BindlessBufferMutable(packed_float3, bindless.Buffer, view.PosedMorphNormalDeltaSlot);
        deltas[draw.PosedPositionOffset + i] = packed_float3(morph_normal_delta);
    }
}

#endif
