#ifndef EDITSELECTION_MSL
#define EDITSELECTION_MSL

#include "ConnectivityRead.metal"
#include "gpu/EditSelectionSummary.h"

inline bool EditSelectionBit(const thread Scene &scene, SlotOffset range, uint address) {
    if (range.Slot == InvalidSlot) return false;
    const uint word = BindlessBuffer(uint, scene.B.Buffer, range.Slot)[(range.Offset * 32u + address) >> 5u];
    return ((word >> (address & 31u)) & 1u) != 0u;
}

inline uint EditElementAddress(const thread Scene &scene, DrawData draw, Element element, uint handle) {
    return element == Element::Vertex ? scene.VertexAddress(draw, handle) : element == Element::Edge ? scene.EdgeAddress(draw, handle) :
                                                                                                          scene.FaceAddress(draw, handle);
}

// Visibility masks apply only while editing this mesh.
inline bool EditElementHidden(const thread Scene &scene,DrawData draw,Element element,uint handle) {
    if (scene.View.InteractionMode!=InteractionMode::Edit || handle==InvalidOffset) return false;
    const uint slot=element==Element::Vertex ? draw.Selection.VertexHiddenSlot : element==Element::Edge ? draw.Selection.EdgeHiddenSlot : draw.Selection.FaceHiddenSlot;
    return slot!=InvalidSlot && EditSelectionBit(scene,{slot,0u},EditElementAddress(scene,draw,element,handle));
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
    for (const auto item : conn.Fan(vertex_id)) {
        const uint h = item.x;
        if (summary.Mode == Element::Face) {
            if (conn.HalfedgeFace(h) == summary.ActiveHandle) return true;
        } else if (conn.IncomingEdge(h) == summary.ActiveHandle || conn.BoundaryOutgoingEdge(h) == summary.ActiveHandle) return true;
    }
    return false;
}

inline uint EditVertexState(const thread Scene &scene, DrawData draw, uint vertex_id) {
    if (draw.Selection.Summary.Slot == InvalidSlot) return 0u;
    const EditSelectionSummary summary = EditSelectionInfo(scene, draw);
    return (EditSelectionBit(scene, draw.Selection.VertexBits, scene.VertexAddress(draw, vertex_id)) ? STATE_SELECTED : 0u) |
        (EditVertexTouchesActive(scene, draw, vertex_id, summary) ? STATE_ACTIVE : 0u);
}

inline ConnectivityView EditConnectivity(const thread Scene &scene, DrawData draw) {
    return {scene.B, draw.Connectivity, draw.FaceCount};
}

inline bool EditEdgeTouchesActiveFace(const thread Scene &scene, DrawData draw, uint edge, uint active_face) {
    if (draw.FaceCount == 0u) return false;
    const auto conn = EditConnectivity(scene, draw);
    const uint halfedge = conn.EdgeHalfedge(edge);
    if (conn.HalfedgeFace(halfedge) == active_face) return true;
    const uint opposite = conn.Opposite(halfedge);
    return opposite != InvalidOffset && conn.HalfedgeFace(opposite) == active_face;
}

inline uint EditEdgeEndpointState(const thread Scene &scene, DrawData draw, uint edge, uint vertex_id) {
    if (draw.Selection.Summary.Slot == InvalidSlot) return 0u;
    const EditSelectionSummary summary = EditSelectionInfo(scene, draw);
    const bool vertex_mode = summary.Mode == Element::Vertex;
    const bool selected = EditSelectionBit(scene, vertex_mode ? draw.Selection.VertexBits : draw.Selection.EdgeBits, vertex_mode ? scene.VertexAddress(draw, vertex_id) : scene.EdgeAddress(draw, edge));
    bool active = summary.Mode == Element::Edge && summary.ActiveHandle == edge;
    if (summary.Mode == Element::Face && summary.ActiveHandle != InvalidOffset) {
        active = EditEdgeTouchesActiveFace(scene, draw, edge, summary.ActiveHandle);
    }
    return (selected ? STATE_SELECTED : 0u) | (active ? STATE_ACTIVE : 0u);
}

inline uint EditFaceState(const thread Scene &scene, DrawData draw, uint face) {
    if (draw.Selection.Summary.Slot == InvalidSlot) return 0u;
    const EditSelectionSummary summary = EditSelectionInfo(scene, draw);
    return (EditSelectionBit(scene, draw.Selection.FaceBits, scene.FaceAddress(draw, face)) ? STATE_SELECTED : 0u) |
        (summary.Mode == Element::Face && summary.ActiveHandle == face ? STATE_ACTIVE : 0u);
}

#endif
