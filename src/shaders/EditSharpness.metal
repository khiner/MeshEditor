#ifndef EDITSHARPNESS_MSL
#define EDITSHARPNESS_MSL

#include "Bindless.metal"
#include "ConnectivityRead.metal"
#include "gpu/EditSharpnessOperation.h"
#include "gpu/EditSharpnessPushConstants.h"


struct EditSharpnessContext {
    device const BindlessSet &B;
    constant EditSharpnessPushConstants &Pc;

    bool Selected(SlotOffset range, uint element) const {
        const uint word = BindlessBuffer(uint, B.Buffer, range.Slot)[range.Offset + (element >> 5u)];
        return ((word >> (element & 31u)) & 1u) != 0u;
    }
    ConnectivityView Connectivity() const {
        return {BindlessBuffer(uint, B.Buffer, Pc.Connectivity.Slot) + Pc.Connectivity.Offset, Pc.VertexCount, Pc.HalfedgeCount, Pc.FaceCount, Pc.ConnectivityFaceStarts != 0u};
    }
};

kernel void EditSharpnessKernel(
    uint i [[thread_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant EditSharpnessPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const EditSharpnessContext ctx{bindless, pc};
    if (i < pc.FaceCount) {
        bool write = false;
        uint value = pc.Value;
        if (pc.Operation == EditSharpnessOperation::SetAllFaces) write = true;
        else if (pc.Operation == EditSharpnessOperation::SmoothAll || pc.Operation == EditSharpnessOperation::SmoothByAngle) {
            write = true;
            value = 0u;
        } else if (pc.Operation == EditSharpnessOperation::SetSelectedFaces) {
            write = ctx.Selected(pc.FaceSelectionBits, i);
        }
        if (write) BindlessBufferMutable(uchar, bindless.Buffer, pc.FaceSharpness.Slot)[pc.FaceSharpness.Offset + i] = uchar(value);
    }

    if (i >= pc.EdgeCount) return;
    bool write_edge = false;
    uint edge_value = pc.Value;
    if (pc.Operation == EditSharpnessOperation::SmoothAll) {
        write_edge = true;
        edge_value = 0u;
    } else if (pc.Operation == EditSharpnessOperation::SetSelectedEdges) {
        write_edge = ctx.Selected(pc.EdgeSelectionBits, i);
    } else if (pc.Operation == EditSharpnessOperation::SetVertexEdges) {
        device const uint *edge_indices = BindlessBuffer(uint, bindless.IndexBuffer, pc.EdgeIndices.Slot) + pc.EdgeIndices.Offset;
        write_edge = ctx.Selected(pc.VertexSelectionBits, edge_indices[i * 2u]) ||
            ctx.Selected(pc.VertexSelectionBits, edge_indices[i * 2u + 1u]);
    } else if (pc.Operation == EditSharpnessOperation::SmoothByAngle) {
        write_edge = true;
        edge_value = 0u;
        const auto conn = ctx.Connectivity();
        const uint h = conn.EdgeHalfedge(i);
        const uint opposite = conn.Opposite(h);
        if (opposite != InvalidOffset) {
            const uint f0 = conn.HalfedgeFace(h), f1 = conn.HalfedgeFace(opposite);
            device const packed_float3 *normals = BindlessBuffer(packed_float3, bindless.Buffer, pc.FaceNormals.Slot) + pc.FaceNormals.Offset;
            edge_value = dot(float3(normals[f0]), float3(normals[f1])) < pc.CosAngle ? 1u : 0u;
        }
    }
    if (write_edge) BindlessBufferMutable(uchar, bindless.Buffer, pc.EdgeSharpness.Slot)[pc.EdgeSharpness.Offset + i] = uchar(edge_value);
}

#endif
