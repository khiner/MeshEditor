#include "ElementWorkShared.metal"
#include "EnclosingSphere.metal"
#include "gpu/MeshletBoundsRefitPushConstants.h"
#include "gpu/Vertex.h"

kernel void MeshletBoundsRefit(
    uint i [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant MeshletBoundsRefitPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    if (i >= pc.Count) return;
    const uint id = WorkGroupElement(b,pc.Work,i);
    if (id == InvalidOffset) return;
    device MeshletRecord *records = BindlessBufferMutable(MeshletRecord,b.Buffer,pc.MeshletSlot);
    MeshletRecord record = records[id];
    if (!record.VertexCount) return;
    device const uint *vertices = BindlessBuffer(uint,b.Buffer,pc.MeshletVertexSlot)+record.VertexOffset;
    device const uchar *local = BindlessBuffer(uchar,b.Buffer,pc.LocalTrianglesSlot)+record.LocalTriangleOffset;
    FitClusterBounds(record,local,[&](uint v) {
        const uint vertex_id = record.Topology == 0u ? BindlessBuffer(uint,b.IndexBuffer,pc.CornerSlot)[vertices[v]] : pc.VertexOffset+vertices[v];
        return float3(BindlessBuffer(Vertex,b.VertexBuffer,pc.VertexSlot)[vertex_id].Position);
    });
    records[id] = record;
}
