#ifndef COMMITPOSEDGEOMETRY_MSL
#define COMMITPOSEDGEOMETRY_MSL

#include "Bindless.metal"
#include "CommitPosedGeometryPushConstants.metal"
#include "ElementWorkShared.metal"
#include "TransformUtils.metal"
#include "FanItemEncoding.metal"
#include "CornerClass.metal"
#include "CornerClassEncoding.metal"

kernel void GeometryWorkArgsKernel(
    uint i [[thread_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant CommitPosedGeometryPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    if (i == 0u) FinishWork(bindless, pc.ChangedVertices);
    if (i == 1u) FinishWork(bindless, pc.Faces);
    if (i == 2u) FinishWork(bindless, pc.Normals);
    if (i == 3u) FinishWork(bindless, pc.Meshlets);
    if (i == 4u) FinishWork(bindless, pc.BoundsTiles);
}

kernel void CommitPosedGeometryKernel(
    uint invocation [[thread_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant CommitPosedGeometryPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    if (pc.Phase == 0u) {
        const uint i = WorkElement(bindless, pc.Candidates, invocation);
        if (i == INVALID_OFFSET) return;
        device Vertex *vertices = BindlessBufferMutable(Vertex, bindless.VertexBuffer, pc.Vertices.Slot) + pc.Vertices.Offset;
        const bool selected = pc.Selection.Slot != INVALID_SLOT &&
            (BindlessBuffer(uint, bindless.Buffer, pc.Selection.Slot)[pc.Selection.Offset + i / 32u] & (1u << (i % 32u))) != 0u;
        if (pc.Commit != 0u && !selected) return;
        const float3 base = float3(vertices[i].Position);
        const float3 world = trs_transform_point(pc.Primary, base);
        const float3 posed = pc.ApplyTransform != 0u && selected ? trs_inverse_transform_point(pc.Primary, apply_edit_transform(world, pc.Pivot, pc.Delta)) : base;
        if (pc.Commit != 0u) {
            if (all(base == posed)) return;
            vertices[i].Position = packed_float3(posed);
        } else {
            device packed_float3 *output = BindlessBufferMutable(packed_float3, bindless.Buffer, pc.Output.Slot) + pc.Output.Offset;
            if (all(float3(output[i]) == posed)) return;
            output[i] = packed_float3(posed);
        }
        MarkWork(bindless, pc.ChangedVertices, i);
        MarkWork(bindless, pc.BoundsTiles, i / 256u);
        if (pc.Entry.FaceCount == 0u) {
            if (pc.TriangleMeshlets.Slot == INVALID_SLOT) return;
            device const uint *map = BindlessBuffer(uint, bindless.Buffer, pc.TriangleMeshlets.Slot) + pc.TriangleMeshlets.Offset;
            if (pc.Topology == 2u) MarkWork(bindless, pc.Meshlets, map[i]);
            else if (pc.VertexEdgeAdjacencyOffset != INVALID_OFFSET) {
                device const uint *edges = BindlessBuffer(uint, bindless.Buffer, pc.AdjacencySlot) + pc.VertexEdgeAdjacencyOffset;
                for (uint j = edges[i]; j < edges[i + 1u]; ++j)
                    MarkWork(bindless, pc.Meshlets, map[edges[pc.Entry.VertexCount + 1u + j]]);
            }
            return;
        }
        device const uint *fans = BindlessBuffer(uint, bindless.Buffer, pc.AdjacencySlot) + pc.Entry.VertexAdjacencyOffset;
        for (uint j = fans[i]; j < fans[i + 1u]; ++j)
            MarkWork(bindless, pc.Faces, fans[pc.Entry.VertexCount + 1u + j] & FanItemEncoding_FaceMask);
    } else if (pc.Phase == 1u) {
        const uint f = WorkElement(bindless, pc.Faces, invocation);
        if (f == INVALID_OFFSET) return;
        device const uint *first = BindlessBuffer(uint, bindless.ObjectIdBuffer, pc.FaceFirstTriangleSlot) + pc.Entry.FaceDataOffset;
        const uint end = f + 1u < pc.Entry.FaceCount ? first[f + 1u] : pc.Entry.TriangleCount;
        device const uint *indices = BindlessBuffer(uint, bindless.IndexBuffer, pc.Entry.FaceIndices.Slot) + pc.Entry.FaceIndices.Offset;
        for (uint t = first[f]; t < end; ++t) {
            if (pc.TriangleMeshlets.Slot != INVALID_SLOT)
                MarkWork(bindless, pc.Meshlets, BindlessBuffer(uint, bindless.Buffer, pc.TriangleMeshlets.Slot)[pc.TriangleMeshlets.Offset + t]);
            for (uint c = 0u; c < 3u; ++c) MarkWork(bindless, pc.Normals, indices[t * 3u + c]);
        }
    } else {
        const uint v = WorkElement(bindless, pc.Normals, invocation);
        if (v >= pc.Entry.VertexCount || pc.CornerClassOffset == INVALID_OFFSET || pc.CornerClassOffset == CornerClassEncoding_UniformFaceOffset) return;
        device const uint *fans = BindlessBuffer(uint, bindless.Buffer, pc.AdjacencySlot) + pc.Entry.VertexAdjacencyOffset;
        device const uint *first = BindlessBuffer(uint, bindless.ObjectIdBuffer, pc.FaceFirstTriangleSlot) + pc.Entry.FaceDataOffset;
        // Sectors are stored per corner: equivalent sectors on other incident faces also need updating.
        for (uint j = fans[v]; j < fans[v + 1u]; ++j) {
            const uint item = fans[pc.Entry.VertexCount + 1u + j];
            const uint f = item & FanItemEncoding_FaceMask, k = item >> FanItemEncoding_LoopShift;
            const uint corner = k < 2u ? first[f] * 3u + k : (first[f] + k - 2u) * 3u + 2u;
            const uint value = BindlessBuffer(uint, bindless.Buffer, pc.CornerClassSlot)[pc.CornerClassOffset + corner];
            if ((value >> CornerClassEncoding_TagShift) == CornerClass_Seam)
                MarkWork(bindless, pc.Normals, pc.Entry.VertexCount + (value & CornerClassEncoding_IndexMask));
        }
    }
}

#endif
