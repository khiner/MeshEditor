#ifndef EDITSHARPNESS_MSL
#define EDITSHARPNESS_MSL

#include "Bindless.metal"
#include "ConnectivityRead.metal"
#include "ElementWorkShared.metal"
#include "gpu/EditSharpnessPushConstants.h"

struct EditSharpnessContext {
    device const BindlessSet &B;
    constant EditSharpnessPushConstants &Pc;

    ConnectivityView Connectivity() const { return {B, Pc.Connectivity, Pc.FaceCount}; }
    bool VertexSelected(uint vertex_id) const {
        return (BindlessBuffer(uint, B.Buffer, Pc.VertexSelectionSlot)[vertex_id >> 5u] & (1u << (vertex_id & 31u))) != 0u;
    }
    uint Selected(SlotOffset at, uint i) const { return BindlessBuffer(uint, B.Buffer, at.Slot)[at.Offset + i]; }
    void WriteFace(uint face, uint value) const { BindlessBufferMutable(uchar, B.Buffer, Pc.FaceSharpnessSlot)[face] = uchar(value); }
    void WriteEdge(uint edge, uint value) const { BindlessBufferMutable(uchar, B.Buffer, Pc.EdgeSharpnessSlot)[edge] = uchar(value); }
    void WriteVertexEdge(uint vertex_id, uint edge) const {
        if (edge == InvalidOffset) return;
        const auto conn = Connectivity();
        const uint h = conn.EdgeHalfedge(edge);
        device const uint *corners = BindlessBuffer(uint, B.IndexBuffer, Pc.CornersSlot);
        const uint a = corners[h], b = corners[conn.Previous(h)];
        const uint other = a == vertex_id ? b : a;
        // Adjacent selected vertices share an edge.
        // Its lower selected endpoint owns the byte write.
        // No concurrent stores to the same edge occur.
        if (other < vertex_id && VertexSelected(other)) return;
        WriteEdge(edge, Pc.Value);
    }
};

kernel void EditSharpnessKernel(
    uint i [[thread_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant EditSharpnessPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const EditSharpnessContext ctx{bindless, pc};
    if (pc.Operation == EditSharpnessOperation::SetSelectedFaces || pc.Operation == EditSharpnessOperation::SetSelectedEdges ||
        pc.Operation == EditSharpnessOperation::SetVertexEdges) {
        if (i >= pc.SelectedCount) return;
        const uint element = ctx.Selected(pc.Selected, i);
        if (pc.Operation == EditSharpnessOperation::SetSelectedFaces) ctx.WriteFace(element, pc.Value);
        else if (pc.Operation == EditSharpnessOperation::SetSelectedEdges) ctx.WriteEdge(element, pc.Value);
        else {
            const auto conn = ctx.Connectivity();
            conn.ForEachIncidentEdge(element, [&](uint edge) { ctx.WriteVertexEdge(element, edge); });
        }
        return;
    }
    if (i < pc.FaceCount) ctx.WriteFace(WorkGroupElement(bindless,pc.FaceWork,i),
        pc.Operation == EditSharpnessOperation::SetAllFaces ? pc.Value : 0u);
    if (i >= pc.EdgeCount || pc.Operation == EditSharpnessOperation::SetAllFaces) return;
    uint value = 0u;
    const uint edge = WorkGroupElement(bindless,pc.EdgeWork,i);
    if (pc.Operation == EditSharpnessOperation::SmoothByAngle) {
        const auto conn = ctx.Connectivity();
        const uint h = conn.EdgeHalfedge(edge), opposite = conn.Opposite(h);
        if (opposite != InvalidOffset) {
            device const packed_float3 *normals = BindlessBuffer(packed_float3, bindless.Buffer, pc.FaceNormalsSlot);
            value = dot(float3(normals[conn.HalfedgeFace(h)]), float3(normals[conn.HalfedgeFace(opposite)])) < pc.CosAngle ? 1u : 0u;
        }
    }
    ctx.WriteEdge(edge, value);
}

#endif
