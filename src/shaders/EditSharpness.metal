#pragma once
#include "ConnectivityRead.metal"
#include "ElementWorkShared.metal"
#include "gpu/EditSharpnessJob.h"

struct EditSharpnessContext {
    device const BindlessSet &B;
    constant EditSharpnessJob &Pc;

    ConnectivityView Connectivity() const { return {B, Pc.Connectivity}; }
    bool VertexSelected(uint vertex_id) const {
        return WorkRank(B, Pc.VertexSelection, vertex_id) != InvalidOffset;
    }
    uint Selected(SlotOffset at, uint i) const { return BindlessBuffer(uint, B.Buffer, at.Slot)[at.Offset + i]; }
    void WriteFace(uint face, uint value) const { BindlessBufferMutable(uchar, B.Buffer, Pc.FaceSharpnessSlot)[Connectivity().FaceAddress(face)] = uchar(value); }
    void WriteEdge(uint edge, uint value) const { BindlessBufferMutable(uchar, B.Buffer, Pc.EdgeSharpnessSlot)[Connectivity().EdgeAddress(edge)] = uchar(value); }
    void WriteVertexEdge(uint vertex_id, uint edge) const {
        if (edge == InvalidOffset) return;
        const auto conn = Connectivity();
        const uint other = conn.OtherEnd(conn.EdgeHalfedge(edge), vertex_id);
        // Only the lower endpoint of an edge between selected vertices writes its byte, so no two lanes store to the same edge.
        if (other < vertex_id && VertexSelected(other)) return;
        WriteEdge(edge, Pc.Value);
    }
};

kernel void EditSharpnessKernel(JOB_TABLE_ARGS(EditSharpnessJob)) {
    JOB_THREAD(pc, i);
    const EditSharpnessContext ctx{b, pc};
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
    if (i < pc.FaceCount) ctx.WriteFace(WorkGroupElement(b,pc.FaceWork,i),
        pc.Operation == EditSharpnessOperation::SetAllFaces ? pc.Value : 0u);
    if (i >= pc.EdgeCount || pc.Operation == EditSharpnessOperation::SetAllFaces) return;
    uint value = 0u;
    const uint edge = WorkGroupElement(b,pc.EdgeWork,i);
    if (pc.Operation == EditSharpnessOperation::SmoothByAngle) {
        const auto conn = ctx.Connectivity();
        const uint2 faces = conn.EdgeFaces(edge);
        if (faces.x != InvalidOffset && faces.y != InvalidOffset) {
            device const packed_float3 *normals = BindlessBuffer(packed_float3, b.Buffer, pc.FaceNormalsSlot);
            value = dot(float3(normals[conn.FaceAddress(faces.x)]), float3(normals[conn.FaceAddress(faces.y)])) < pc.CosAngle ? 1u : 0u;
        }
    }
    ctx.WriteEdge(edge, value);
}
