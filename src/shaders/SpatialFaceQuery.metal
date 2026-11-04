#include "ElementWorkShared.metal"
#include "ScreenSpace.metal"
#include "gpu/PlaneSide.h"
#include "gpu/SpatialFaceQueryPushConstants.h"

inline bool SpatialFaceAffected(device const BindlessSet &b, constant SpatialFaceQueryPushConstants &pc, uint face) {
    device const uint *ranges = BindlessBuffer(uint, b.Buffer, pc.FaceRangeSlot);
    const ElementValues<const uint> corners{BindlessBuffer(uint, b.IndexBuffer, pc.CornerSlot), BindlessBuffer(uint, b.Buffer, pc.Tables.Halfedges.Slot) + pc.Tables.Halfedges.Offset};
    const ElementValues<const Vertex> vertices{BindlessBuffer(Vertex, b.VertexBuffer, pc.VertexSlot), BindlessBuffer(uint, b.Buffer, pc.Tables.Vertices.Slot) + pc.Tables.Vertices.Offset};
    const uint address = HandleAddress<MeshElementBlockSize>(b, pc.Tables.Faces, face);
    const uint first = ranges[2u * address], last = ranges[2u * address + 1u];
    if (pc.Mode == 0u) {
        bool negative = false, positive = false, snapped = false;
        for (uint h = first; h < last; ++h) {
            const float3 p = float3(vertices[corners[h]].Position);
            const float distance = dot(float3(pc.PlaneNormal), p) - pc.PlaneOffset;
            const PlaneSideKind side = PlaneSideOf(distance, p, PlaneSideTolerance);
            negative |= side == PlaneSideKind::Negative;
            positive |= side == PlaneSideKind::Positive;
            snapped |= side == PlaneSideKind::On && distance != 0.f;
        }
        return (negative && positive) || snapped;
    }
    const float4x4 to_clip = pc.ScreenTransform.Unpack();
    for (uint h = first; h < last; ++h) {
        const uint from = corners[h == first ? last - 1u : h - 1u], to = corners[h];
        if (KnifeEdgeParameter(to_clip, float2(pc.Extent), float3(vertices[from].Position), float3(vertices[to].Position), float2(pc.KnifeStart), float2(pc.KnifeEnd)) > 0.f) return true;
    }
    return false;
}

// Each candidate face the query affects marks itself, without render ownership.
kernel void SpatialFaceQueryExpand(device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant SpatialFaceQueryPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= pc.CandidateCount) return;
    const uint face = WorkGroupElement(b, pc.Candidates, index);
    if (SpatialFaceAffected(b, pc, face)) MarkWork(b, pc.Faces, face);
}
