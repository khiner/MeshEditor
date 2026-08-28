#ifndef COMMITPOSEDGEOMETRY_MSL
#define COMMITPOSEDGEOMETRY_MSL

#include "Bindless.metal"
#include "CommitPosedGeometryPushConstants.metal"

kernel void CommitPosedGeometryKernel(
    uint i [[thread_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant CommitPosedGeometryPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    if (i < pc.VertexCount) {
        device Vertex *vertices = BindlessBufferMutable(Vertex, bindless.VertexBuffer, pc.Vertices.Slot) + pc.Vertices.Offset;
        const packed_float3 posed = BindlessBuffer(packed_float3, bindless.Buffer, pc.PosedPositions.Slot)[pc.PosedPositions.Offset + i];
        if (any(float3(vertices[i].Position) != float3(posed))) {
            vertices[i].Position = posed;
            device atomic_uint *changed = BindlessBufferMutable(atomic_uint, bindless.Buffer, pc.Changed.Slot) + pc.Changed.Offset;
            atomic_store_explicit(changed, 1u, memory_order_relaxed);
        }
        if (pc.PosedVertexNormals.Slot != INVALID_SLOT) {
            BindlessBufferMutable(packed_float3, bindless.Buffer, pc.BaseVertexNormals.Slot)[pc.BaseVertexNormals.Offset + i] =
                BindlessBuffer(packed_float3, bindless.Buffer, pc.PosedVertexNormals.Slot)[pc.PosedVertexNormals.Offset + i];
        }
    }
    if (i < pc.SeamCount) {
        BindlessBufferMutable(packed_float3, bindless.Buffer, pc.BaseSeamNormals.Slot)[pc.BaseSeamNormals.Offset + i] =
            BindlessBuffer(packed_float3, bindless.Buffer, pc.PosedSeamNormals.Slot)[pc.PosedSeamNormals.Offset + i];
    }
    if (i < pc.FaceCount) {
        BindlessBufferMutable(packed_float3, bindless.Buffer, pc.BaseFaceNormals.Slot)[pc.BaseFaceNormals.Offset + i] =
            BindlessBuffer(packed_float3, bindless.Buffer, pc.PosedFaceNormals.Slot)[pc.PosedFaceNormals.Offset + i];
    }
}

#endif
