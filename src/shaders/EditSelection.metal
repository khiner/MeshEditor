#ifndef EDITSELECTION_MSL
#define EDITSELECTION_MSL

#include "Bindless.metal"
#include "ConnectivityRead.metal"
#include "gpu/EditSelectionSummary.h"
#include "gpu/Element.h"
#include "gpu/FanItemEncoding.h"

inline bool EditSelectionBit(const thread Scene &scene, SlotOffset range, uint element) {
    if (range.Slot == InvalidSlot) return false;
    const uint word = BindlessBuffer(uint, scene.B.Buffer, range.Slot)[range.Offset + (element >> 5u)];
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
    const uint adjacency_offset = summary.Mode == Element::Edge ? draw.VertexEdgeAdjacencyOffset : draw.VertexFanAdjacencyOffset;
    if (adjacency_offset == InvalidOffset || scene.View.AdjacencySlot == InvalidSlot) return false;
    device const uint *adjacency = scene.Adjacency(scene.View.AdjacencySlot) + adjacency_offset;
    const uint item_base = draw.VertexCountOrHeadImageSlot + 1u;
    for (uint i = adjacency[vertex_id]; i < adjacency[vertex_id + 1u]; ++i) {
        const uint item = adjacency[item_base + i];
        const uint handle = summary.Mode == Element::Face ? item & uint(FanItemEncoding::FaceMask) : item;
        if (handle == summary.ActiveHandle) return true;
    }
    return false;
}

inline uint EditVertexState(const thread Scene &scene, DrawData draw, uint vertex_id) {
    if (draw.Selection.Summary.Slot == InvalidSlot) return 0u;
    const EditSelectionSummary summary = EditSelectionInfo(scene, draw);
    return (EditSelectionBit(scene, draw.Selection.VertexBits, vertex_id) ? STATE_SELECTED : 0u) |
        (EditVertexTouchesActive(scene, draw, vertex_id, summary) ? STATE_ACTIVE : 0u);
}

inline uint EditHalfedgeFace(const thread Scene &scene, DrawData draw, uint halfedge) {
    device const uint *connectivity = BindlessBuffer(uint, scene.B.Buffer, draw.Connectivity.Slot) + draw.Connectivity.Offset;
    return ConnectivityHalfedgeFace(
        connectivity, draw.VertexCountOrHeadImageSlot, draw.HalfedgeCount,
        draw.FaceCount, draw.ConnectivityFaceStarts != 0u, halfedge
    );
}

inline bool EditEdgeTouchesActiveFace(const thread Scene &scene, DrawData draw, uint edge, uint active_face) {
    if (draw.Connectivity.Slot == InvalidSlot || draw.EdgeHalfedges.Slot == InvalidSlot) return false;
    device const uint *connectivity = BindlessBuffer(uint, scene.B.Buffer, draw.Connectivity.Slot) + draw.Connectivity.Offset;
    device const uint *opposites = ConnectivityOpposites(connectivity, draw.VertexCountOrHeadImageSlot);
    const uint halfedge = BindlessBuffer(uint, scene.B.Buffer, draw.EdgeHalfedges.Slot)[draw.EdgeHalfedges.Offset + edge];
    if (EditHalfedgeFace(scene, draw, halfedge) == active_face) return true;
    const uint opposite = opposites[halfedge];
    return opposite != InvalidOffset && EditHalfedgeFace(scene, draw, opposite) == active_face;
}

inline uint EditEdgeEndpointState(const thread Scene &scene, DrawData draw, uint edge, uint vertex_id) {
    if (draw.Selection.Summary.Slot == InvalidSlot) return 0u;
    const EditSelectionSummary summary = EditSelectionInfo(scene, draw);
    const bool vertex_mode = summary.Mode == Element::Vertex;
    const bool selected = EditSelectionBit(
        scene, vertex_mode ? draw.Selection.VertexBits : draw.Selection.EdgeBits,
        vertex_mode ? vertex_id : edge
    );
    bool active = summary.Mode == Element::Edge && summary.ActiveHandle == edge;
    if (summary.Mode == Element::Face && summary.ActiveHandle != InvalidOffset) {
        active = EditEdgeTouchesActiveFace(scene, draw, edge, summary.ActiveHandle);
    }
    return (selected ? STATE_SELECTED : 0u) | (active ? STATE_ACTIVE : 0u);
}

inline uint EditFaceState(const thread Scene &scene, DrawData draw, uint face) {
    if (draw.Selection.Summary.Slot == InvalidSlot) return 0u;
    const EditSelectionSummary summary = EditSelectionInfo(scene, draw);
    return (EditSelectionBit(scene, draw.Selection.FaceBits, face) ? STATE_SELECTED : 0u) |
        (summary.Mode == Element::Face && summary.ActiveHandle == face ? STATE_ACTIVE : 0u);
}

#endif
