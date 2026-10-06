#ifndef VERTEXBLOCKS_MSL
#define VERTEXBLOCKS_MSL

#include "Bindless.metal"
#include "Frustum.metal"
#include "MeshletShared.metal"
#include "PyramidOcclusion.metal"
#include "TransformUtils.metal"
#include "VertexBounds.metal"
#include "gpu/InstanceRecord.h"
#include "gpu/MeshElementBlock.h"
#include "gpu/SelectionAggregate.h"
#include "gpu/VertexBlockPushConstants.h"
#include "EditSelection.metal"

constant uint VertexBlockLanes = MeshElementBlockSize / VertexBlockGroups;
constant uint VertexBlockSimdGroups = VertexBlockLanes / 32u;

struct VertexBlockLane {
    InstanceRecord Instance;
    DrawData Draw;
    uint ActiveVertex; // The mesh's active vertex in Excite mode.
    // The lane's live vertex relative to the draw's vertex origin, InvalidOffset for a dead slot or a culled block.
    uint VertexId;
};

// Whether the block's live positions, placed by the instance, can reach the view.
// A block without known bounds, or any instance during an object transform, stays visible.
// The occlusion test grows the block's footprint by `margin` pixels and pulls its depth by `depth_pull` in clip space, as the drawn primitives are.
inline bool VertexBlockVisible(
    const thread Scene &scene, constant VertexBlockPushConstants &pc, uint block, Transform world, float margin, float depth_pull
) {
    AABB bounds;
    if (pc.LeafSlot != InvalidSlot) bounds = BindlessBuffer(SelectionAggregate, scene.B.Buffer, pc.LeafSlot)[block].Bounds;
    else {
        const uint at = VertexBoundsFind(scene.B, pc.BoundsNodesSlot, pc.BoundsMembersSlot, pc.BoundsNamespace, VertexBoundsKey(0u, block));
        if (at == InvalidOffset) return true;
        bounds = BindlessBuffer(AABB, scene.B.Buffer, pc.BoundsValuesSlot)[at];
    }
    if (scene.View.IsTransforming != 0u && scene.View.InteractionMode != InteractionMode::Edit) return true;
    const OrientedBounds box = TransformBounds(bounds, world);
    if (!box.Valid) return true;
    if (!in_frustum(scene.ViewProj(), box.Center, box.Ax, box.Ay, box.Az)) return false;
    if (pc.MinDiameterPixels > 0.0f &&
        ProjectedDiameterPixels(scene, box.Center, length(box.Ax) + length(box.Ay) + length(box.Az)) < pc.MinDiameterPixels) return false;
    return pc.PyramidSamplerSlot == InvalidSlot ||
        !BoxPastPyramid(scene, pc.PyramidSamplerSlot, box.Center, box.Ax, box.Ay, box.Az, margin, depth_pull);
}

// The threadgroup's quarter of its block, with each lane gated by the block's live mask and the block's visibility.
// The first lane of each SIMD group tests the block's visibility for the group.
inline VertexBlockLane ResolveVertexBlockLane(
    const thread Scene &scene, constant VertexBlockPushConstants &pc, uint group, uint lane, float margin = 0.0f, float depth_pull = 0.0f
) {
    const uint block = BindlessBuffer(uint, scene.B.Buffer, pc.Blocks.Slot)[pc.Blocks.Offset + group / VertexBlockGroups];
    const InstanceRecord instance = scene.InstanceRecords(scene.View.InstanceRecordSlot)[pc.Instance];
    const MeshRecord mesh = scene.MeshRecords(scene.View.MeshRecordSlot)[instance.Mesh];
    const DrawData draw = ComposeDraw(mesh, instance, pc.Instance);
    uint visible = 0u;
    if (simd_is_first()) visible = VertexBlockVisible(scene, pc, block, MeshletWorld(scene, draw), margin, depth_pull) ? 1u : 0u;
    visible = simd_broadcast_first(visible);
    const uint slot = (group % VertexBlockGroups) * VertexBlockLanes + lane;
    const uint live = BindlessBuffer(MeshElementBlock, scene.B.Buffer, pc.MembershipSlot)[block].Live[slot / 32u];
    const bool present = visible != 0u && (live & (1u << (slot % 32u))) != 0u &&
        !EditElementHidden(scene,draw,Element::Vertex,block*MeshElementBlockSize+slot);
    return {instance, draw, mesh.Display.ActiveVertex, present ? block * MeshElementBlockSize + slot - draw.VertexOffset : InvalidOffset};
}

#endif
