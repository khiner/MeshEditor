#include "ElementWorkShared.metal"
#include "EnclosingSphere.metal"
#include "gpu/MeshletBoundsJob.h"

kernel void MeshletBoundsRefit(JOB_TABLE_ARGS(MeshletBoundsJob)) {
    JOB_TILE_THREAD(job, i, MeshletBoundsTile);
    if (i >= job.Count) return;
    const uint address = WorkGroupElement(b,job.MeshletAddresses,i);
    if (address == InvalidOffset) return;
    device MeshletRecord *records = BindlessBufferMutable(MeshletRecord,b.Buffer,job.MeshletSlot);
    MeshletRecord record = records[address];
    if (!record.VertexCount) return;
    device const uint *vertices = BindlessBuffer(uint,b.Buffer,job.MeshletVertexSlot)+job.VertexBase+record.VertexOffset;
    device const uchar *local = BindlessBuffer(uchar,b.Buffer,job.LocalTrianglesSlot)+(job.LocalTriangleBase+record.LocalTriangleOffset)*3u;
    FitClusterBounds(record,local,[&](uint v) {
        const uint handle = record.Topology == 0u ? BindlessBuffer(uint,b.IndexBuffer,job.CornerSlot)[HandleAddress<MeshElementBlockSize>(b,job.CornerTable,vertices[v])] : vertices[v];
        return float3(BindlessBuffer(Vertex,b.VertexBuffer,job.VertexSlot)[HandleAddress<MeshElementBlockSize>(b,job.VertexTable,handle)].Position);
    });
    records[address] = record;
}
