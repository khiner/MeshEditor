#ifndef COMMITPOSEDGEOMETRY_MSL
#define COMMITPOSEDGEOMETRY_MSL

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

inline void MarkOwnedMeshlet(device const BindlessSet &b, constant CommitPosedGeometryPushConstants &pc, device const uint *meshlets, uint topology, uint handle) {
    const auto map=pc.ElementMeshlets[topology];
    if (map.Slot == InvalidSlot) return;
    const uint owner = BindlessBuffer(uint,b.Buffer,map.Slot)[map.Offset+handle];
    if (owner != InvalidOffset) MarkWork(b,pc.MeshletAddresses,Resolve<RenderBlockSize>(meshlets,owner));
}

kernel void CommitPosedGeometryKernel(
    uint invocation [[thread_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant CommitPosedGeometryPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    if (pc.Phase == 4u) {
        const uint i = WorkElement(bindless, pc.Candidates, invocation);
        if (i == InvalidOffset) return;
        if (pc.Mode == GeometryEditMode::Commit) return;
        if (pc.Mode == GeometryEditMode::Preview) {
            const uint address = HandleAddress<MeshElementBlockSize>(bindless, pc.Entry.Connectivity.Tables.Vertices, i);
            const float3 position = float3(BindlessBuffer(Vertex, bindless.VertexBuffer, pc.VerticesSlot)[address].Position);
            const uint destination = PoseAttributeIndex(bindless, pc.PositionNodesSlot, pc.Entry.PositionNamespace, address);
            device packed_float3 *output = BindlessBufferMutable(packed_float3, bindless.Buffer, pc.PositionSlot);
            if (all(float3(output[destination]) == position)) return;
            output[destination] = packed_float3(position);
        }
        MarkWork(bindless, pc.ChangedVertices, i);
    } else if (pc.Phase == 1u) {
        const uint i = WorkElement(bindless, pc.Candidates, invocation);
        if (i == InvalidOffset) return;
        const ConnectivityView conn{bindless, pc.Entry.Connectivity, pc.Entry.FaceCount};
        const auto fan=conn.Fan(i);
        const uint address = conn.VertexAddress(i);
        MarkWork(bindless, pc.BoundsTiles, address / 256u);
        device const uint *meshlets = SetTable(bindless, pc.MeshletSets, pc.MeshletSet);
        if (!fan.Count) MarkOwnedMeshlet(bindless,pc,meshlets,2u,i);
        for (const auto item : fan) {
            if (item.y==InvalidOffset) MarkOwnedMeshlet(bindless,pc,meshlets,1u,conn.Edge(item.x));
            else MarkWork(bindless, pc.Faces, item.y);
        }
    } else {
        const uint f = WorkElement(bindless, pc.Faces, invocation);
        if (f == InvalidOffset) return;
        device const uint *triangles = BindlessBuffer(uint, bindless.ObjectIdBuffer, pc.FaceTriangleStartSlot);
        const ConnectivityView conn{bindless, pc.Entry.Connectivity, pc.Entry.FaceCount};
        const uint2 loop = conn.FaceHalfedges(f);
        const uint first = triangles[conn.FaceAddress(f)], end = first + loop.y - loop.x - 2u;
        device const uint *corners = BindlessBuffer(uint, bindless.IndexBuffer, pc.Entry.CornersSlot);
        device const uint *meshlets = SetTable(bindless, pc.MeshletSets, pc.MeshletSet);
        for (uint t = first; t < end; ++t) MarkOwnedMeshlet(bindless,pc,meshlets,0u,t);
        for (uint h = loop.x; h < loop.y; ++h) MarkWork(bindless, pc.Normals, corners[conn.HalfedgeAddress(h)]);
    }
}

#endif
