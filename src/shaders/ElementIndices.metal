#ifndef ELEMENTINDICES_MSL
#define ELEMENTINDICES_MSL

// Writes each edge's from and to vertex in edge order, and each fan triangle's corners in triangle order.
#include "Bindless.metal"
#include "BlockScan.metal"
#include "ConnectivityRead.metal"
#include "gpu/ElementIndicesJob.h"
#include "gpu/TiledJobPushConstants.h"

kernel void EdgeEndpointsWrite(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant TiledJobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const uint2 tile = BindlessBuffer(uint2, bindless.Buffer, pc.TileMapSlot)[pc.FirstTile + group_id];
    const ElementIndicesJob job = BindlessBuffer(ElementIndicesJob, bindless.Buffer, pc.JobsSlot)[tile.x];
    const uint e = tile.y * ScanTileSize + lane;
    if (e >= job.EdgeCount) return;
    const ConnectivityView conn{BindlessBuffer(uint, bindless.Buffer, job.Connectivity.Slot) + job.Connectivity.Offset, job.VertexCount, job.HalfedgeCount, job.FaceCount, job.FaceStarts != 0u};
    const uint h = conn.EdgeHalfedge(e);
    device const uint *corners = BindlessBuffer(uint, bindless.IndexBuffer, job.Corners.Slot) + job.Corners.Offset;
    device uint *endpoints = BindlessBufferMutable(uint, bindless.IndexBuffer, job.Endpoints.Slot) + job.Endpoints.Offset;
    endpoints[2u * e] = corners[conn.Previous(h)];
    endpoints[2u * e + 1u] = corners[h];
}

// Triangle t is fan triangle k of its face, spanning the face's first corner and corners k + 1 and k + 2.
kernel void TriangleIndicesWrite(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant TiledJobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const uint2 tile = BindlessBuffer(uint2, bindless.Buffer, pc.TileMapSlot)[pc.FirstTile + group_id];
    const ElementIndicesJob job = BindlessBuffer(ElementIndicesJob, bindless.Buffer, pc.JobsSlot)[tile.x];
    const uint t = tile.y * ScanTileSize + lane;
    if (t >= job.TriangleCount) return;
    const ConnectivityView conn{BindlessBuffer(uint, bindless.Buffer, job.Connectivity.Slot) + job.Connectivity.Offset, job.VertexCount, job.HalfedgeCount, job.FaceCount, job.FaceStarts != 0u};
    const uint f = BindlessBuffer(uint, bindless.ObjectIdBuffer, job.TriangleFaceIds.Slot)[job.TriangleFaceIds.Offset + t] - 1u;
    const uint k = t - BindlessBuffer(uint, bindless.ObjectIdBuffer, job.FaceFirstTriangles.Slot)[job.FaceFirstTriangles.Offset + f];
    const uint start = conn.FaceHalfedges(f).x;
    device const uint *corners = BindlessBuffer(uint, bindless.IndexBuffer, job.Corners.Slot) + job.Corners.Offset;
    device uint *indices = BindlessBufferMutable(uint, bindless.IndexBuffer, job.Triangles.Slot) + job.Triangles.Offset;
    indices[3u * t] = corners[start];
    indices[3u * t + 1u] = corners[start + k + 1u];
    indices[3u * t + 2u] = corners[start + k + 2u];
}

#endif
