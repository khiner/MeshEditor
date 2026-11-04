#pragma once
#include "Bindless.metal"
#include "gpu/ElementWork.h"

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

// The live element at thread i over a domain's blocks, each block's slots one thread each, or InvalidOffset.
inline uint BlockHandle(uint i, device const BindlessSet &b, DomainBlocks blocks) {
    const uint k = i / MeshElementBlockSize, lane = i % MeshElementBlockSize;
    if (k >= blocks.Count) return InvalidOffset;
    const uint block = BindlessBuffer(uint, b.Buffer, blocks.Blocks.Slot)[blocks.Blocks.Offset + k];
    device const MeshElementBlock &membership = BindlessBuffer(MeshElementBlock, b.Buffer, blocks.MembershipSlot)[block];
    if (!SlotLive(membership, lane)) return InvalidOffset;
    return membership.Local * MeshElementBlockSize + lane;
}

inline uint LowerBound(device const uint *values, uint count, uint value) {
    return PartitionPoint(count, [&](uint k) { return values[k] < value; });
}
inline bool SortedContains(device const uint *values, uint count, uint value) {
    const uint k = LowerBound(values, count, value);
    return k < count && values[k] == value;
}

// All arguments and stored references are record-relative handles, which the record's tables resolve to arena addresses.
struct ConnectivityView {
    device const BindlessSet &B;
    ConnectivityRef At;
    device const uint *VertexTable, *HalfedgeTable, *EdgeTable, *FaceTable;
    device const uint *Corners, *Opposites, *HalfedgeEdges, *HalfedgeFaces, *EdgeHalfedges;
    device const packed_uint2 *FaceRanges;

    ConnectivityView(device const BindlessSet &b, ConnectivityRef at)
        : B(b), At(at), VertexTable(TableRun(b, at.Tables.Vertices)), HalfedgeTable(TableRun(b, at.Tables.Halfedges)),
          EdgeTable(TableRun(b, at.Tables.Edges)), FaceTable(TableRun(b, at.Tables.Faces)),
          Corners(at.CornersSlot == InvalidSlot ? nullptr : BindlessBuffer(uint, b.IndexBuffer, at.CornersSlot)), Opposites(Stream<uint>(b, at.OppositesSlot)),
          HalfedgeEdges(Stream<uint>(b, at.HalfedgeEdgesSlot)), HalfedgeFaces(Stream<uint>(b, at.HalfedgeFacesSlot)), EdgeHalfedges(Stream<uint>(b, at.EdgesSlot)),
          FaceRanges(Stream<packed_uint2>(b, at.FaceRangesSlot)) {}
    static device const uint *TableRun(device const BindlessSet &b, SlotOffset table) { return table.Slot == InvalidSlot ? nullptr : BindlessBuffer(uint, b.Buffer, table.Slot) + table.Offset; }
    template<typename T> static device const T *Stream(device const BindlessSet &b, uint slot) { return slot == InvalidSlot ? nullptr : BindlessBuffer(T, b.Buffer, slot); }

    uint VertexAddress(uint v) const { return Resolve<MeshElementBlockSize>(VertexTable, v); }
    uint HalfedgeAddress(uint h) const { return Resolve<MeshElementBlockSize>(HalfedgeTable, h); }
    uint EdgeAddress(uint e) const { return Resolve<MeshElementBlockSize>(EdgeTable, e); }
    uint FaceAddress(uint f) const { return Resolve<MeshElementBlockSize>(FaceTable, f); }
    uint VertexOf(uint h) const { return Corners[HalfedgeAddress(h)]; }
    uint From(uint h) const { return VertexOf(Previous(h)); }

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
    // The halfedge of a fan item's face along the edge from the fan's vertex to c, running either way, or InvalidOffset when the item has no face or no such side.
    uint FanSide(uint2 item, uint c) const {
        if (item.y == InvalidOffset) return InvalidOffset;
        const uint next = Next(item.x, item.y);
        if (VertexOf(next) == c) return next;
        return VertexOf(Previous(item.x)) == c ? item.x : InvalidOffset;
    }
    // The face sides between a and c, counted at the corners of a.
    uint FaceSides(uint a, uint c) const {
        uint sides = 0u;
        for (const auto item : Fan(a)) sides += FanSide(item, c) != InvalidOffset ? 1u : 0u;
        return sides;
    }
    uint2 EdgeEnds(uint e) const {
        const uint h = EdgeHalfedge(e);
        return uint2(From(h), VertexOf(h));
    }
    // The end of halfedge h's edge other than v, or v when both ends are v.
    uint OtherEnd(uint h, uint v) const {
        const uint a = From(h);
        return a == v ? VertexOf(h) : a;
    }
    uint2 EdgeFaces(uint e) const {
        const uint h = EdgeHalfedge(e), o = Opposite(h);
        return uint2(HalfedgeFace(h), o == InvalidOffset ? InvalidOffset : HalfedgeFace(o));
    }
    uint EdgeFaceCount(uint e) const {
        const uint2 faces = EdgeFaces(e);
        return (faces.x != InvalidOffset ? 1u : 0u) + (faces.y != InvalidOffset ? 1u : 0u);
    }
    bool FaceHas(uint f, uint v) const {
        const uint2 range = FaceHalfedges(f);
        for (uint h = range.x; h < range.y; ++h) {
            if (VertexOf(h) == v) return true;
        }
        return false;
    }
    uint Valence(uint f) const {
        const uint2 range = FaceHalfedges(f);
        return range.y - range.x;
    }
    // The face sides along an edge, counted at the origin of its first halfedge.
    uint EdgeSides(uint e) const {
        const uint2 ends = EdgeEnds(e);
        return FaceSides(ends.x, ends.y);
    }
    // The halfedge of a face other than `face` along the edge of corner h, running either way, or InvalidOffset.
    uint OtherSide(uint h, uint face) const {
        const uint c = VertexOf(h);
        for (const auto item : Fan(From(h))) {
            if (item.y == face) continue;
            if (const uint side = FanSide(item, c); side != InvalidOffset) return side;
        }
        return InvalidOffset;
    }

    // Visits each representative incoming edge and unpaired outgoing edge of the vertex once.
    template<typename Visit>
    void ForEachIncidentEdge(uint vertex_handle, Visit visit) const {
        for (const auto item : Fan(vertex_handle)) {
            if (const uint edge = IncomingEdge(item.x); edge != InvalidOffset) visit(edge);
            if (const uint edge = BoundaryOutgoingEdge(item.x, item.y); edge != InvalidOffset) visit(edge);
        }
    }
    // The edge joining a and c, the last one found around a, or InvalidOffset.
    uint EdgeBetween(uint a, uint c) const {
        uint found = InvalidOffset;
        ForEachIncidentEdge(a, [&](uint e) {
            const uint h = EdgeHalfedge(e);
            if (From(h) == c || VertexOf(h) == c) found = e;
        });
        return found;
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
