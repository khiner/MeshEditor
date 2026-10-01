#ifndef POSEPREPASS_MSL
#define POSEPREPASS_MSL

// Writes canonical posed positions and reduces their vertex-block bounds.
#include "Bindless.metal"
#include "MorphDeform.metal"
#include "ArmatureDeform.metal"
#include "TransformUtils.metal"
#include "gpu/BoundsReducePushConstants.h"
#include "EditSelection.metal"
#include "BoundsShared.metal"
#include "VertexBounds.metal"

kernel void PosePrepassKernel(
    uint local_id [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    threadgroup float3 *shared_min [[threadgroup(0)]],
    threadgroup float3 *shared_max [[threadgroup(1)]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant BoundsReducePushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    const uint2 tile = VertexBoundsTile(bindless,pc,group_id);
    const BoundsEntry entry = scene.BoundsEntries(pc.BoundsEntrySlot)[tile.x];
    const DrawData draw = scene.BoundsDraw(entry);
    const uint vertex_id = tile.y*256u+local_id;
    float3 lo=AabbEmptyMin, hi=AabbEmptyMax;
    if (BoundsVertexLive(bindless,entry,tile.y,local_id)) {
        const uint i=vertex_id-draw.VertexOffset;
        float3 pos = float3(scene.Vertices(draw.VertexSlot)[vertex_id].Position);
        float3 normal = float3(0);
        float3 morph_normal_delta = float3(0);
        ApplyMorphDeform(scene, draw, pos, morph_normal_delta, i);
        pos = ApplyArmatureDeform(scene, draw, pos, i, normal);
        if (view.IsTransforming != 0u && draw.HasPendingVertexTransform != 0u &&
            EditSelectionBit(scene, draw.Selection.VertexBits, i)) {
            const Transform primary = scene.Models(draw.ModelSlot)[draw.PrimaryEditInstanceIndex];
            pos = trs_inverse_transform_point(primary, apply_pending_transform_world(scene, trs_transform_point(primary, pos)));
        }
        if (draw.PositionNamespace != InvalidOffset) {
            const uint at=PoseAttributeIndex(bindless,view.PosedPositionNodesSlot,draw.PositionNamespace,vertex_id);
            BindlessBufferMutable(packed_float3,bindless.Buffer,view.PosedPositionSlot)[at]=packed_float3(pos);
        }
        if (draw.MorphShadingAuthored != 0u) {
            const uint at=PoseAttributeIndex(bindless,view.PosedMorphNormalNodesSlot,draw.MorphNormalNamespace,vertex_id);
            BindlessBufferMutable(packed_float3,bindless.Buffer,view.PosedMorphNormalDeltaSlot)[at]=packed_float3(morph_normal_delta);
        }
        lo=pos; hi=pos;
    }
    FoldSharedAabb(shared_min,shared_max,BoundsFoldLanes,local_id,lo,hi);
    if (local_id == 0u) {
        const uint destination=VertexBoundsIndex(bindless,pc,entry,0u,tile.y);
        BindlessBufferMutable(AABB,bindless.Buffer,pc.ValuesSlot)[destination]={packed_float3(shared_min[0]),packed_float3(shared_max[0])};
    }
}
#endif
