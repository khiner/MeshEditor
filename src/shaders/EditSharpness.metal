#ifndef EDITSHARPNESS_MSL
#define EDITSHARPNESS_MSL

#include "Bindless.metal"
#include "ConnectivityRead.metal"
#include "EditSharpnessOperation.metal"
#include "EditSharpnessPushConstants.metal"

constant uint EditSharpnessInvalid = 0xffffffffu;

struct EditSharpnessContext {
    device const BindlessSet &B;
    constant EditSharpnessPushConstants &Pc;

    bool Selected(SlotOffset range, uint element) const {
        const uint word = BindlessBuffer(uint, B.Buffer, range.Slot)[range.Offset + (element >> 5u)];
        return ((word >> (element & 31u)) & 1u) != 0u;
    }
    device const uint *Connectivity() const { return BindlessBuffer(uint, B.Buffer, Pc.Connectivity.Slot) + Pc.Connectivity.Offset; }
    device const uint *Opposites() const { return ConnectivityOpposites(Connectivity(), Pc.VertexCount); }
    uint HalfedgeFace(uint halfedge) const {
        return ConnectivityHalfedgeFace(Connectivity(), Pc.VertexCount, Pc.HalfedgeCount, Pc.FaceCount, Pc.ConnectivityFaceStarts != 0u, halfedge);
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
        if (pc.Operation == EditSharpnessOperation_SetAllFaces) write = true;
        else if (pc.Operation == EditSharpnessOperation_SmoothAll || pc.Operation == EditSharpnessOperation_SmoothByAngle) {
            write = true;
            value = 0u;
        } else if (pc.Operation == EditSharpnessOperation_SetSelectedFaces) {
            write = ctx.Selected(pc.FaceSelectionBits, i);
        }
        if (write) BindlessBufferMutable(uchar, bindless.Buffer, pc.FaceSharpness.Slot)[pc.FaceSharpness.Offset + i] = uchar(value);
    }

    if (i >= pc.EdgeCount) return;
    bool write_edge = false;
    uint edge_value = pc.Value;
    if (pc.Operation == EditSharpnessOperation_SmoothAll) {
        write_edge = true;
        edge_value = 0u;
    } else if (pc.Operation == EditSharpnessOperation_SetSelectedEdges) {
        write_edge = ctx.Selected(pc.EdgeSelectionBits, i);
    } else if (pc.Operation == EditSharpnessOperation_SetVertexEdges) {
        device const uint *edge_indices = BindlessBuffer(uint, bindless.IndexBuffer, pc.EdgeIndices.Slot) + pc.EdgeIndices.Offset;
        write_edge = ctx.Selected(pc.VertexSelectionBits, edge_indices[i * 2u]) ||
            ctx.Selected(pc.VertexSelectionBits, edge_indices[i * 2u + 1u]);
    } else if (pc.Operation == EditSharpnessOperation_SmoothByAngle) {
        write_edge = true;
        edge_value = 0u;
        const uint h = BindlessBuffer(uint, bindless.Buffer, pc.EdgeHalfedges.Slot)[pc.EdgeHalfedges.Offset + i];
        const uint opposite = ctx.Opposites()[h];
        if (opposite != EditSharpnessInvalid) {
            const uint f0 = ctx.HalfedgeFace(h), f1 = ctx.HalfedgeFace(opposite);
            device const packed_float3 *normals = BindlessBuffer(packed_float3, bindless.Buffer, pc.FaceNormals.Slot) + pc.FaceNormals.Offset;
            edge_value = dot(float3(normals[f0]), float3(normals[f1])) < pc.CosAngle ? 1u : 0u;
        }
    }
    if (write_edge) BindlessBufferMutable(uchar, bindless.Buffer, pc.EdgeSharpness.Slot)[pc.EdgeSharpness.Offset + i] = uchar(edge_value);
}

#endif
