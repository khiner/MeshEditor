#ifndef EDITSELECTION_MSL
#define EDITSELECTION_MSL

#include "Bindless.metal"
#include "ConnectivityRead.metal"
#include "gpu/EditSelectionSummary.h"
#include "gpu/Element.h"

inline bool EditSelectionBit(const thread Scene &scene, SlotOffset range, uint element) {
    if (range.Slot == InvalidSlot) return false;
    const uint word = BindlessBuffer(uint, scene.B.Buffer, range.Slot)[(range.Offset * 32u + element) >> 5u];
    return ((word >> (element & 31u)) & 1u) != 0u;
}

inline EditSelectionSummary EditSelectionInfo(const thread Scene &scene, DrawData draw) {
    if (draw.Selection.Summary.Slot == InvalidSlot) {
        return {.ActiveHandle = InvalidOffset};
    }
    return BindlessBuffer(EditSelectionSummary, scene.B.Buffer, draw.Selection.Summary.Slot)[draw.Selection.Summary.Offset];
}

inline bool EditVertexTouchesActive(const thread Scene &scene, DrawData draw, uint vertex_id, EditSelectionSummary summary) {
    if (summary.ActiveHandle == InvalidOffset) return false;
    if (summary.Mode == Element::Vertex) return vertex_id == summary.ActiveHandle;
    const ConnectivityView conn{scene.B, draw.Connectivity, draw.FaceCount};
    for (const auto item : conn.Fan(draw.VertexOffset + vertex_id)) {
        const uint h = item.x;
        if (summary.Mode == Element::Face) {
            if (conn.FaceOrdinal(conn.HalfedgeFace(h)) == summary.ActiveHandle) return true;
        } else if (conn.EdgeOrdinal(conn.IncomingEdge(h)) == summary.ActiveHandle || conn.EdgeOrdinal(conn.BoundaryOutgoingEdge(h)) == summary.ActiveHandle) return true;
    }
    return false;
}

inline uint EditVertexState(const thread Scene &scene, DrawData draw, uint vertex_id) {
    if (draw.Selection.Summary.Slot == InvalidSlot) return 0u;
    const EditSelectionSummary summary = EditSelectionInfo(scene, draw);
    return (EditSelectionBit(scene, draw.Selection.VertexBits, vertex_id) ? STATE_SELECTED : 0u) |
        (EditVertexTouchesActive(scene, draw, vertex_id, summary) ? STATE_ACTIVE : 0u);
}

inline ConnectivityView EditConnectivity(const thread Scene &scene, DrawData draw) {
    return {scene.B, draw.Connectivity, draw.FaceCount};
}

inline bool EditEdgeTouchesActiveFace(const thread Scene &scene, DrawData draw, uint edge, uint active_face) {
    if (draw.FaceCount == 0u) return false;
    const auto conn = EditConnectivity(scene, draw);
    const uint halfedge = conn.EdgeHalfedge(draw.Connectivity.Edges.Offset + edge);
    if (conn.FaceOrdinal(conn.HalfedgeFace(halfedge)) == active_face) return true;
    const uint opposite = conn.Opposite(halfedge);
    return opposite != InvalidOffset && conn.FaceOrdinal(conn.HalfedgeFace(opposite)) == active_face;
}

inline uint EditEdgeEndpointState(const thread Scene &scene, DrawData draw, uint edge, uint vertex_id) {
    if (draw.Selection.Summary.Slot == InvalidSlot) return 0u;
    const EditSelectionSummary summary = EditSelectionInfo(scene, draw);
    const bool vertex_mode = summary.Mode == Element::Vertex;
    const bool selected = EditSelectionBit(scene, vertex_mode ? draw.Selection.VertexBits : draw.Selection.EdgeBits, vertex_mode ? vertex_id : edge);
    bool active = summary.Mode == Element::Edge && summary.ActiveHandle == edge;
    if (summary.Mode == Element::Face && summary.ActiveHandle != InvalidOffset) {
        active = EditEdgeTouchesActiveFace(scene, draw, edge, summary.ActiveHandle);
    }
    return (selected ? STATE_SELECTED : 0u) | (active ? STATE_ACTIVE : 0u);
}

inline uint EditFaceState(const thread Scene &scene, DrawData draw, uint face) {
    if (draw.Selection.Summary.Slot == InvalidSlot) return 0u;
    // Draw IDs are relative to the stable origin.
    // Mask addresses are canonical.
    face -= draw.Connectivity.FaceRanges.Offset;
    const EditSelectionSummary summary = EditSelectionInfo(scene, draw);
    return (EditSelectionBit(scene, draw.Selection.FaceBits, face) ? STATE_SELECTED : 0u) |
        (summary.Mode == Element::Face && summary.ActiveHandle == face ? STATE_ACTIVE : 0u);
}

#endif
