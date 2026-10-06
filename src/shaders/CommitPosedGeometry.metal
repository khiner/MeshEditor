#ifndef COMMITPOSEDGEOMETRY_MSL
#define COMMITPOSEDGEOMETRY_MSL

#include "Bindless.metal"
#include "gpu/CommitPosedGeometryPushConstants.h"
#include "ElementWorkShared.metal"
#include "ConnectivityRead.metal"
#include "TransformUtils.metal"

kernel void FinalizeElementWorkKernel(
    uint tid [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant ElementWork *work [[buffer(BufferIndex_PushConstants)]]
) {
    threadgroup uint totals[8];
    FinishWork(bindless, work[group], tid, totals);
}

// Counts emissions before the host reserves sparse block tables, one add per SIMD group.
// The host bounds each table by the blocks its domain's set spans, even when many vertices share the same faces.
inline void CountWorkBlocks(device const BindlessSet &b, constant CommitPosedGeometryPushConstants &pc, uint field, uint amount) {
    const uint total = simd_sum(amount);
    if (!simd_is_first()) return;
    device atomic_uint *count = BindlessBufferMutable(atomic_uint, b.Buffer, pc.Candidates.Storage.Slot) + pc.BudgetOffset + field;
    atomic_fetch_add_explicit(count, total, memory_order_relaxed);
}

inline void MarkOwnedMeshlet(device const BindlessSet &b, constant CommitPosedGeometryPushConstants &pc, uint topology, uint element) {
    const auto map=pc.ElementMeshlets[topology];
    if (map.ValuesSlot == InvalidSlot) return;
    const uint block = BindlessBuffer(uint,b.Buffer,map.BlocksSlot)[element/256u];
    if (!block) return;
    const uint owner = BindlessBuffer(uint,b.Buffer,map.ValuesSlot)[(block-1u)*256u+element%256u];
    if (owner != InvalidOffset) MarkWork(b,pc.Meshlets,owner);
}

kernel void CommitPosedGeometryKernel(
    uint invocation [[thread_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant CommitPosedGeometryPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    if (pc.Phase == 4u) {
        const uint i = WorkElement(bindless, pc.Candidates, invocation);
        if (i == InvalidOffset) return;
        if (pc.Mode != GeometryEditMode::Refresh) {
            device Vertex *vertices = BindlessBufferMutable(Vertex, bindless.VertexBuffer, pc.Vertices.Slot);
            const bool selected = pc.SelectionSlot != InvalidSlot &&
                (BindlessBuffer(uint, bindless.Buffer, pc.SelectionSlot)[i / 32u] & (1u << (i % 32u))) != 0u;
            if (pc.Mode == GeometryEditMode::Commit && !selected) return;
            const float3 base = float3(vertices[i].Position);
            const float3 posed = pc.ApplyTransform != 0u && selected ? apply_geometry_edit_transform(base, pc.Primary, pc.Pivot, pc.Delta) : base;
            if (pc.Mode == GeometryEditMode::Commit) {
                if (all(base == posed)) return;
                vertices[i].Position = packed_float3(posed);
            } else {
                const uint destination = PoseAttributeIndex(bindless, pc.PositionNodesSlot, pc.Entry.PositionNamespace, i);
                device packed_float3 *output = BindlessBufferMutable(packed_float3, bindless.Buffer, pc.PositionSlot);
                if (all(float3(output[destination]) == posed)) return;
                output[destination] = packed_float3(posed);
            }
        }
        MarkWork(bindless, pc.ChangedVertices, i);
    } else if (pc.Phase < 2u) {
        const uint i = WorkElement(bindless, pc.Candidates, invocation);
        if (i == InvalidOffset) return;
        const ConnectivityView conn{bindless, pc.Entry.Connectivity, pc.Entry.FaceCount};
        const auto fan=conn.Fan(i);
        if (pc.Phase == 0u) {
            uint faces=0u, wires=0u;
            for (const auto item : fan) {
                if (item.y==InvalidOffset) ++wires;
                else ++faces;
            }
            CountWorkBlocks(bindless, pc, 0u, faces);
            CountWorkBlocks(bindless, pc, 2u, fan.Count ? wires : 1u);
            return;
        }
        MarkWork(bindless, pc.BoundsTiles, i / 256u);
        if (!fan.Count) MarkOwnedMeshlet(bindless,pc,2u,i);
        for (const auto item : fan) {
            if (item.y==InvalidOffset) MarkOwnedMeshlet(bindless,pc,1u,conn.Edge(item.x));
            else MarkWork(bindless, pc.Faces, item.y);
        }
    } else {
        const uint f = WorkElement(bindless, pc.Faces, invocation);
        if (f == InvalidOffset) return;
        device const uint *triangles = BindlessBuffer(uint, bindless.ObjectIdBuffer, pc.FaceTriangleStartSlot);
        const ConnectivityView conn{bindless, pc.Entry.Connectivity, pc.Entry.FaceCount};
        const uint2 loop = conn.FaceHalfedges(f);
        if (pc.Phase == 2u) {
            CountWorkBlocks(bindless, pc, 1u, loop.y - loop.x);
            CountWorkBlocks(bindless, pc, 2u, loop.y - loop.x - 2u);
            return;
        }
        const uint first = triangles[f], end = first + loop.y - loop.x - 2u;
        device const uint *corners = BindlessBuffer(uint, bindless.IndexBuffer, pc.Entry.Corners.Slot);
        for (uint t = first; t < end; ++t) MarkOwnedMeshlet(bindless,pc,0u,t);
        for (uint h = loop.x; h < loop.y; ++h) MarkWork(bindless, pc.Normals, corners[h]);
    }
}

#endif
