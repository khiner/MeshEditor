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

// All arguments and stored references are canonical arena handles.
struct ConnectivityView {
    device const BindlessSet &B;
    ConnectivityRef At;
    uint FaceCount;
    uint EdgeOrdinal(uint e) const { return e == InvalidOffset ? e : e - At.Edges.Offset; }
    uint FaceOrdinal(uint f) const { return f == InvalidOffset ? f : f - At.FaceRanges.Offset; }

    uint2 Incoming(uint v) const { if (At.VertexCorners.Slot == InvalidSlot) return uint2(InvalidOffset, 0u); return uint2(BindlessBuffer(packed_uint2,B.Buffer,At.VertexCorners.Slot)[v]); }
    uint2 FanItem(uint index) const { return uint2(BindlessBuffer(packed_uint2,B.Buffer,At.FanItemsSlot)[index]); }
    uint FanCorner(uint index) const { return FanItem(index).x; }
    ConnectivityFan Fan(uint v) const {
        const uint2 range = Incoming(v);
        if (!range.y) return {};
        return {BindlessBuffer(packed_uint2,B.Buffer,At.FanItemsSlot),range.x,range.y};
    }
    uint Next(uint h, uint face) const {
        if (FaceCount == 0u || face == InvalidOffset) return InvalidOffset;
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

    device const uint *Words(SlotOffset at) const { return BindlessBuffer(uint,B.Buffer,at.Slot); }
    device const uint *Outgoing() const { return Words(At.Outgoing); }
    device const uint *Opposites() const { return Words(At.Opposites); }
    device const uint *HalfedgeToEdge() const { return Words(At.HalfedgeEdges); }
    device const uint *HalfedgeToFace() const { return Words(At.HalfedgeFaces); }
    device const packed_uint2 *Ranges() const { return BindlessBuffer(packed_uint2,B.Buffer,At.FaceRanges.Slot); }
    device const uint *Edges() const { return Words(At.Edges); }

    uint Opposite(uint h) const { return Opposites()[h]; }
    uint Edge(uint h) const { return HalfedgeToEdge()[h]; }
    uint EdgeHalfedge(uint e) const { return Edges()[e]; }
    bool EdgeFirst(uint h) const { return EdgeHalfedge(Edge(h)) == h; }
    uint2 FaceHalfedges(uint f) const { return uint2(Ranges()[f]); }
    uint HalfedgeFace(uint h) const { return FaceCount == 0u ? InvalidOffset : HalfedgeToFace()[h]; }
    // A line corner's previous corner is its pair, the corner at the line's other end.
    uint Previous(uint h) const {
        const uint face = HalfedgeFace(h);
        if (face == InvalidOffset) return Opposite(h);
        const uint2 range = FaceHalfedges(face);
        return h == range.x ? range.y - 1u : h - 1u;
    }
};
#endif
