#ifndef CONNECTIVITYREAD_MSL
#define CONNECTIVITYREAD_MSL

#include "Bindless.metal"
#include "gpu/ConnectivityRef.h"

struct ConnectivityFan {
    device const packed_uint2 *Items;
    uint First,Count;
    struct Iterator {
        device const packed_uint2 *Items;
        uint Index;
        uint2 operator*() const { return uint2(Items[Index]); }
        thread Iterator &operator++() { ++Index; return *this; }
        bool operator!=(Iterator other) const { return Index!=other.Index; }
    };
    Iterator begin() const { return {Items,First}; }
    Iterator end() const { return {Items,First+Count}; }
};

// All arguments and stored references are record-relative handles, which the record's tables resolve to arena addresses.
// The view resolves its table runs and streams once, so each address costs one table read.
struct ConnectivityView {
    device const BindlessSet &B;
    ConnectivityRef At;
    uint FaceCount;
    device const uint *VertexTable, *HalfedgeTable, *EdgeTable, *FaceTable;
    device const uint *Opposites, *HalfedgeEdges, *HalfedgeFaces, *EdgeHalfedges;
    device const packed_uint2 *FaceRanges;

    ConnectivityView(device const BindlessSet &b, ConnectivityRef at, uint face_count)
        : B(b), At(at), FaceCount(face_count), VertexTable(TableRun(b, at.Tables.Vertices)), HalfedgeTable(TableRun(b, at.Tables.Halfedges)),
          EdgeTable(TableRun(b, at.Tables.Edges)), FaceTable(TableRun(b, at.Tables.Faces)), Opposites(Stream<uint>(b, at.OppositesSlot)),
          HalfedgeEdges(Stream<uint>(b, at.HalfedgeEdgesSlot)), HalfedgeFaces(Stream<uint>(b, at.HalfedgeFacesSlot)), EdgeHalfedges(Stream<uint>(b, at.EdgesSlot)),
          FaceRanges(Stream<packed_uint2>(b, at.FaceRangesSlot)) {}
    static device const uint *TableRun(device const BindlessSet &b, SlotOffset table) { return table.Slot == InvalidSlot ? nullptr : BindlessBuffer(uint, b.Buffer, table.Slot) + table.Offset; }
    template<typename T> static device const T *Stream(device const BindlessSet &b, uint slot) { return slot == InvalidSlot ? nullptr : BindlessBuffer(T, b.Buffer, slot); }

    uint VertexAddress(uint v) const { return Resolve<MeshElementBlockSize>(VertexTable, v); }
    uint HalfedgeAddress(uint h) const { return Resolve<MeshElementBlockSize>(HalfedgeTable, h); }
    uint EdgeAddress(uint e) const { return Resolve<MeshElementBlockSize>(EdgeTable, e); }
    uint FaceAddress(uint f) const { return Resolve<MeshElementBlockSize>(FaceTable, f); }
    // A corner's vertex handle, from the corner vertices at `corner_slot` of the index buffers.
    uint CornerVertex(uint corner_slot, uint h) const { return BindlessBuffer(uint, B.IndexBuffer, corner_slot)[HalfedgeAddress(h)]; }

    uint2 Incoming(uint v) const { if (At.VertexCornersSlot == InvalidSlot) return uint2(InvalidOffset, 0u); return uint2(BindlessBuffer(packed_uint2,B.Buffer,At.VertexCornersSlot)[VertexAddress(v)]); }
    uint2 FanItem(uint index) const { return uint2(BindlessBuffer(packed_uint2,B.Buffer,At.FanItemsSlot)[index]); }
    uint FanCorner(uint index) const { return FanItem(index).x; }
    ConnectivityFan Fan(uint v) const {
        const uint2 range = Incoming(v);
        if (!range.y) return {};
        return {BindlessBuffer(packed_uint2,B.Buffer,At.FanItemsSlot),range.x,range.y};
    }
    uint Next(uint h, uint face) const {
        if (face == InvalidOffset) return InvalidOffset;
        const uint2 range = FaceHalfedges(face);
        return h + 1u < range.y ? h + 1u : range.x;
    }
    uint Next(uint h) const { return Next(h, HalfedgeFace(h)); }
    // An edge is emitted once at each endpoint, including nonmanifold input.
    uint IncomingEdge(uint h) const {
        const uint e = Edge(h), first = EdgeHalfedge(e);
        return h == first || h == Opposite(first) ? e : InvalidOffset;
    }
    uint BoundaryOutgoingEdge(uint h, uint face) const {
        const uint next = Next(h, face);
        if (next == InvalidOffset || Opposite(next) != InvalidOffset) return InvalidOffset;
        const uint e = Edge(next);
        return EdgeHalfedge(e) == next ? e : InvalidOffset;
    }
    uint BoundaryOutgoingEdge(uint h) const { return BoundaryOutgoingEdge(h, HalfedgeFace(h)); }

    // Each representative incoming edge and unpaired outgoing edge is visited once.
    template<typename Visit>
    void ForEachIncidentEdge(uint vertex_handle, Visit visit) const {
        for (const auto item : Fan(vertex_handle)) {
            if (const uint edge = IncomingEdge(item.x); edge != InvalidOffset) visit(edge);
            if (const uint edge = BoundaryOutgoingEdge(item.x, item.y); edge != InvalidOffset) visit(edge);
        }
    }

    device const uint *Words(uint slot) const { return BindlessBuffer(uint,B.Buffer,slot); }

    uint Outgoing(uint v) const { return Words(At.OutgoingSlot)[VertexAddress(v)]; }
    uint Opposite(uint h) const { return Opposites[HalfedgeAddress(h)]; }
    uint Edge(uint h) const { return HalfedgeEdges[HalfedgeAddress(h)]; }
    uint EdgeHalfedge(uint e) const { return EdgeHalfedges[EdgeAddress(e)]; }
    bool EdgeFirst(uint h) const { return EdgeHalfedge(Edge(h)) == h; }
    uint2 FaceHalfedges(uint f) const { return uint2(FaceRanges[FaceAddress(f)]); }
    uint HalfedgeFace(uint h) const { return HalfedgeFaces[HalfedgeAddress(h)]; }
    // A line corner's previous corner is its pair, the corner at the line's other end.
    uint Previous(uint h) const {
        const uint face = HalfedgeFace(h);
        if (face == InvalidOffset) return Opposite(h);
        const uint2 range = FaceHalfedges(face);
        return h == range.x ? range.y - 1u : h - 1u;
    }
};
#endif
