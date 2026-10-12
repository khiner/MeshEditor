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
    const uint address = WorkGroupElement(b,pc.MeshletAddresses,i);
    if (address == InvalidOffset) return;
    device MeshletRecord *records = BindlessBufferMutable(MeshletRecord,b.Buffer,pc.MeshletSlot);
    MeshletRecord record = records[address];
    if (!record.VertexCount) return;
    device const uint *vertices = BindlessBuffer(uint,b.Buffer,pc.MeshletVertexSlot)+pc.VertexBase+record.VertexOffset;
    device const uchar *local = BindlessBuffer(uchar,b.Buffer,pc.LocalTrianglesSlot)+(pc.LocalTriangleBase+record.LocalTriangleOffset)*3u;
    FitClusterBounds(record,local,[&](uint v) {
        const uint handle = record.Topology == 0u ? BindlessBuffer(uint,b.IndexBuffer,pc.CornerSlot)[HandleAddress<MeshElementBlockSize>(b,pc.CornerTable,vertices[v])] : vertices[v];
        return float3(BindlessBuffer(Vertex,b.VertexBuffer,pc.VertexSlot)[HandleAddress<MeshElementBlockSize>(b,pc.VertexTable,handle)].Position);
    });
    records[address] = record;
}
