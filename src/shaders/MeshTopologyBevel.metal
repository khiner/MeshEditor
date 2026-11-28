#pragma once
// Bevels selected edges or vertices by Param0 or per-element widths, with Steps profile segments.
// noinline on the functions below keeps the topology kernels fast to compile.
#include "MeshTopologyFaces.metal"


inline bool TopoBevelVertices(device const MeshTopologyJob &job) { return job.Op == MeshTopologyOp::BevelVertices; }
inline uint TopoBevelSegments(device const MeshTopologyJob &job) { return max(job.Steps, 1u); }
// Whether the edge of its representative halfedge `h` lies between exactly two faces that run it opposite ways.
// The faces count at the edge's end with the smaller fan.
inline bool TopoEdgeTwoFaced(TopoContext ctx, device const MeshTopologyJob &job, uint h) {
    if (ctx.SrcOpposite(job, h) == InvalidOffset) return false;
    const auto conn = ctx.Src(job);
    const uint to = conn.VertexOf(h), from = conn.From(h);
    return (conn.Incoming(to).y <= conn.Incoming(from).y ? conn.FaceSides(to, from) : conn.FaceSides(from, to)) == 2u;
}
// A selected edge bevels where it lies between two faces, whose sides its strip joins, which the halfedge marks hold per edge.
// An edge of one face or of more than two stays as it is.
inline bool TopoEdgeBeveled(TopoContext ctx, device const MeshTopologyJob &job, uint h) {
    if (TopoBevelVertices(job)) return false;
    const uint edge = ctx.SrcEdge(job, h);
    return ctx.SrcSelectedEdge(job, edge) && ctx.EdgeParams(job)[edge] != 0u;
}

inline bool TopoVertexBeveled(TopoContext ctx, device const MeshTopologyJob &job, uint v) {
    if (TopoBevelVertices(job)) return ctx.SrcSelectedVertex(job, v);
    const uint2 fan = ctx.SrcFan(job, v);
    for (uint k = 0u; k < fan.y; ++k) {
        const uint h = ctx.SrcFanCorner(job,fan.x+k);
        if (TopoEdgeBeveled(ctx, job, h) || TopoEdgeBeveled(ctx, job, ctx.SrcNext(job, h))) return true;
    }
    return false;
}

// Whether the corner where `h` arrives has its arriving and leaving edges beveled, as (in, out).
inline bool2 TopoCornerBeveled(TopoContext ctx, device const MeshTopologyJob &job, uint h) {
    return bool2(TopoEdgeBeveled(ctx, job, h), TopoEdgeBeveled(ctx, job, ctx.SrcNext(job, h)));
}

// The distance along a corner's other edge where an offset line of its beveled edge meets it.
inline float TopoBevelSlide(float3 v, float3 along, float3 beveled, float width) {
    const float3 a = TopoDirection(along - v), b = TopoDirection(beveled - v);
    const float sine = length(cross(a, b));
    return width / max(sine, 1e-3f);
}

// The direction from end `v` of unbeveled edge `rep` to its point, or false when `v` is not beveled.
// The point lies where the offset line of a beveled edge that shares a corner with `rep` at `v` meets `rep`, at the farther of the two sides' meetings.
__attribute__((noinline)) bool TopoBevelEdgeDirection(TopoContext ctx, device const MeshTopologyJob &job, uint rep, bool at_to, thread float3 &direction) {
    const auto corners = ctx.SrcCorners(job);
    const uint opposite = ctx.SrcOpposite(job, rep);
    const uint v = at_to ? corners[rep] : ctx.SrcFrom(job, rep);
    const uint other = at_to ? ctx.SrcFrom(job, rep) : corners[rep];
    const float3 p = ctx.SrcPosition(job, v), a = ctx.SrcPosition(job, other);
    if (TopoBevelVertices(job)) {
        if (!ctx.SrcSelectedVertex(job, v)) return false;
        direction = TopoDirection(a - p) * TopoUnit0Vertex(ctx, job, v);
        return true;
    }
    float best = -1.f;
    const auto consider = [&](uint corner_halfedge, bool arriving) {
        const uint candidate = arriving ? ctx.SrcNext(job, corner_halfedge) : corner_halfedge;
        if (!TopoEdgeBeveled(ctx, job, candidate)) return;
        const uint far = arriving ? corners[candidate] : ctx.SrcFrom(job, candidate);
        const float3 q = TopoDirection(a - p) * TopoBevelSlide(p, a, ctx.SrcPosition(job, far), TopoUnit0Edge(ctx, job, ctx.SrcEdge(job, candidate)));
        const float d = length(q);
        if (d > best) {
            best = d;
            direction = q;
        }
    };
    if (at_to) {
        consider(rep, true);
        if (opposite != InvalidOffset) consider(ctx.SrcPrev(job, opposite), false);
    } else {
        consider(ctx.SrcPrev(job, rep), false);
        if (opposite != InvalidOffset) consider(opposite, true);
    }
    if (best >= 0.f) return true;
    // An edge that shares no corner with a beveled edge at its beveled vertex slides by the widest of that vertex's beveled edges.
    // Every edge at a beveled vertex then ends at a point of its own on the vertex's ring.
    float width = -1.f;
    const uint2 fan = ctx.SrcFan(job, v);
    for (uint k = 0u; k < fan.y; ++k) {
        const uint h = ctx.SrcFanCorner(job, fan.x + k), next = ctx.SrcNext(job, h);
        if (TopoEdgeBeveled(ctx, job, h)) width = max(width, TopoUnit0Edge(ctx, job, ctx.SrcEdge(job, h)));
        if (TopoEdgeBeveled(ctx, job, next)) width = max(width, TopoUnit0Edge(ctx, job, ctx.SrcEdge(job, next)));
    }
    if (width < 0.f) return false;
    direction = TopoDirection(a - p) * width;
    return true;
}

// A corner between two beveled edges insets by the mean of their widths, along the direction from its vertex.
inline float3 TopoBevelCornerDirection(TopoContext ctx, device const MeshTopologyJob &job, uint h) {
    const auto corners = ctx.SrcCorners(job);
    const uint prev = ctx.SrcPrev(job, h), next = ctx.SrcNext(job, h);
    const float width = 0.5f * (TopoUnit0Edge(ctx, job, ctx.SrcEdge(job, h)) + TopoUnit0Edge(ctx, job, ctx.SrcEdge(job, next)));
    return TopoInsetCorner(ctx.SrcPosition(job, corners[prev]), ctx.SrcPosition(job, corners[h]), ctx.SrcPosition(job, corners[next]), TopoDirection(float3(ctx.SrcFaceNormals(job)[ctx.SrcFaceOf(job, h)])), width, 0.f, true);
}

struct BevelHalfedgeOutputs {
    bool EndFrom, EndTo, Corner;
    uint Profiles; // 2 * (segments - 1) for a beveled representative
    uint Count() const { return (EndFrom ? 1u : 0u) + (EndTo ? 1u : 0u) + (Corner ? 1u : 0u) + Profiles; }
    uint EndIndex(bool at_to) const { return at_to ? (EndFrom ? 1u : 0u) : 0u; }
    uint CornerIndex() const { return (EndFrom ? 1u : 0u) + (EndTo ? 1u : 0u); }
    uint ProfileIndex(bool at_to, uint j, uint segments) const { return CornerIndex() + (Corner ? 1u : 0u) + (at_to ? segments - 1u : 0u) + j; }
};

__attribute__((noinline)) BevelHalfedgeOutputs TopoBevelOutputs(TopoContext ctx, device const MeshTopologyJob &job, uint h) {
    BevelHalfedgeOutputs o{false, false, false, 0u};
    const bool rep = ctx.SrcEdgeFirst(job, h);
    const bool beveled = TopoEdgeBeveled(ctx, job, h);
    float3 direction;
    if (rep && !beveled) {
        o.EndFrom = TopoBevelEdgeDirection(ctx, job, h, false, direction);
        o.EndTo = TopoBevelEdgeDirection(ctx, job, h, true, direction);
    }
    const bool2 both = TopoCornerBeveled(ctx, job, h);
    o.Corner = both.x && both.y;
    if (rep && beveled) o.Profiles = 2u * (TopoBevelSegments(job) - 1u);
    return o;
}

// A point key holds a count entry and a local index, which resolve to an output vertex after the scan.
inline uint2 TopoBevelKey(uint entry, uint local) { return uint2(entry, local); }
inline bool TopoBevelKeyValid(uint2 key) { return key.x != InvalidOffset; }
inline bool TopoBevelKeyEqual(uint2 a, uint2 b) { return all(a == b); }
inline uint TopoBevelVertexOf(TopoContext ctx, device const MeshTopologyJob &job, uint2 key) {
    return TopoBevelKeyValid(key) ? ctx.Counts(job, TopoCountVertices)[key.x] + key.y : InvalidOffset;
}

inline uint TopoBevelVertexRingIndex(TopoContext ctx, device const MeshTopologyJob &job, uint entry, uint2 boundary,
                                   uint index, uint length, uint ring) {
    return ring == 0u ? TopoBevelVertexOf(ctx, job, boundary) :
        ctx.Counts(job, TopoCountVertices)[entry] + (ring - 1u) * length + index;
}

// The halfedge that holds the outputs of `h`'s edge, and whether `h` runs the way it does.
// An edge of more than two faces has several halfedges each way.
inline uint TopoBevelRepresentative(TopoContext ctx, device const MeshTopologyJob &job, uint h, thread bool &same_way) {
    const uint representative = ctx.SrcEdgeHalfedge(job, ctx.SrcEdge(job, h));
    same_way = ctx.SrcCorners(job)[representative] == ctx.SrcCorners(job)[h];
    return representative;
}

// The key of the point on halfedge `h`'s edge at its end `at_to`, or InvalidOffset when the edge has none there.
inline uint2 TopoBevelEdgeVertex(TopoContext ctx, device const MeshTopologyJob &job, uint h, bool at_to) {
    bool same_way;
    const uint representative = TopoBevelRepresentative(ctx, job, h, same_way);
    const bool end = same_way ? at_to : !at_to;
    const BevelHalfedgeOutputs o = TopoBevelOutputs(ctx, job, representative);
    if (end ? !o.EndTo : !o.EndFrom) return uint2(InvalidOffset);
    return TopoBevelKey(ctx.HalfedgeEntry(job, representative), o.EndIndex(end));
}

// The direction from beveled vertex `v` to its ring point.
inline float3 TopoBevelVertexRingDirection(TopoContext ctx, device const MeshTopologyJob &job, uint v, uint2 boundary,
                                         uint source, uint ring) {
    const auto corners = ctx.SrcCorners(job);
    const uint along = TopoBevelKeyEqual(boundary, TopoBevelEdgeVertex(ctx, job, source, true)) ?
        ctx.SrcFrom(job, source) : corners[ctx.SrcNext(job, source)];
    const float3 center = ctx.SrcPosition(job, v);
    const float3 radial = TopoDirection(ctx.SrcPosition(job, along) - center);
    const float t = float(ring) / float(TopoBevelSegments(job));
    const float3 normal = TopoDirection(float3(ctx.SrcVertexNormals(job)[v]));
    const float convex = max(-dot(radial, normal), 0.f);
    const float remaining = 1.f - t;
    return TopoUnit0Vertex(ctx, job, v) * (radial * remaining * remaining - normal * (0.5f * convex * t * t));
}

inline uint2 TopoBevelCornerVertex(TopoContext ctx, device const MeshTopologyJob &job, uint h) {
    return TopoBevelKey(ctx.HalfedgeEntry(job, h), TopoBevelOutputs(ctx, job, h).CornerIndex());
}

// A corner's replacement points: up to two point keys with the corner as their attribute source.
// Edge holds the halfedge whose edge a point lies on, or InvalidOffset for the corner's own point.
struct BevelCornerPoints {
    uint Count;
    uint2 Vertex[2];
    uint Edge[2];
};

__attribute__((noinline)) BevelCornerPoints TopoBevelCornerPoints(TopoContext ctx, device const MeshTopologyJob &job, uint h) {
    BevelCornerPoints result{0u, {uint2(InvalidOffset), uint2(InvalidOffset)}, {InvalidOffset, InvalidOffset}};
    const uint v = ctx.SrcCorners(job)[h];
    if (!TopoVertexBeveled(ctx, job, v)) {
        result.Vertex[result.Count++] = TopoBevelKey(ctx.VertexEntry(v), 0u);
        return result;
    }
    const bool2 both = TopoCornerBeveled(ctx, job, h);
    if (both.x && both.y) {
        result.Vertex[result.Count++] = TopoBevelCornerVertex(ctx, job, h);
        return result;
    }
    const uint next = ctx.SrcNext(job, h);
    const uint2 on_in = both.x ? uint2(InvalidOffset) : TopoBevelEdgeVertex(ctx, job, h, true);
    const uint2 on_out = both.y ? uint2(InvalidOffset) : TopoBevelEdgeVertex(ctx, job, next, false);
    if (TopoBevelKeyValid(on_in)) {
        result.Edge[result.Count] = h;
        result.Vertex[result.Count++] = on_in;
    }
    if (TopoBevelKeyValid(on_out) && !TopoBevelKeyEqual(on_out, on_in)) {
        result.Edge[result.Count] = next;
        result.Vertex[result.Count++] = on_out;
    }
    if (result.Count == 0u && !both.x && !both.y) result.Vertex[result.Count++] = TopoBevelKey(ctx.VertexEntry(v), 0u);
    return result;
}

// The direction from end `at_to` of a beveled edge to its side point in its own halfedge's face.
__attribute__((noinline)) float3 TopoBevelSideDirection(TopoContext ctx, device const MeshTopologyJob &job, uint edge_halfedge, bool at_to) {
    const uint corner = at_to ? edge_halfedge : ctx.SrcPrev(job, edge_halfedge);
    const bool2 both = TopoCornerBeveled(ctx, job, corner);
    if (both.x && both.y) return TopoBevelCornerDirection(ctx, job, corner);
    const uint other = at_to ? ctx.SrcNext(job, corner) : corner;
    bool same_way;
    const uint representative = TopoBevelRepresentative(ctx, job, other, same_way);
    float3 direction = float3(0.f);
    TopoBevelEdgeDirection(ctx, job, representative, same_way ? at_to : !at_to, direction);
    return direction;
}

inline uint2 TopoBevelSideVertex(TopoContext ctx, device const MeshTopologyJob &job, uint edge_halfedge, bool at_to) {
    const uint corner = at_to ? edge_halfedge : ctx.SrcPrev(job, edge_halfedge);
    const BevelCornerPoints points = TopoBevelCornerPoints(ctx, job, corner);
    return at_to ? points.Vertex[0] : points.Vertex[points.Count - 1u];
}

// Walks the corners around vertex `v` in rotation order from an open side, calling `emit(key, source, index)` for each distinct ring point.
struct BevelRingInfo { uint Length, FirstSource; };
template<typename Emit>
inline BevelRingInfo TopoBevelRing(TopoContext ctx, device const MeshTopologyJob &job, uint v, Emit emit) {
    const uint2 fan = ctx.SrcFan(job, v);
    if (fan.y == 0u) return {0u, InvalidOffset};
    uint start = ctx.SrcFanCorner(job,fan.x);
    uint back = start;
    for (uint step = 0u; step < fan.y; ++step) {
        const uint opposite = ctx.SrcOpposite(job, back);
        if (opposite == InvalidOffset) break;
        const uint previous = ctx.SrcPrev(job, opposite);
        if (previous == start) break;
        back = previous;
    }
    start = back;
    const uint segments = TopoBevelSegments(job);
    uint length = 0u, first_source = InvalidOffset;
    uint2 previous_vertex = uint2(InvalidOffset), first_vertex = uint2(InvalidOffset);
    const auto push = [&](uint2 id, uint source) {
        if (!TopoBevelKeyValid(id) || TopoBevelKeyEqual(id, previous_vertex)) return;
        if (length == 0u) { first_vertex = id; first_source = source; }
        emit(id, source, length);
        ++length;
        previous_vertex = id;
    };
    uint h = start;
    for (uint step = 0u; step < fan.y; ++step) {
        const BevelCornerPoints points = TopoBevelCornerPoints(ctx, job, h);
        for (uint i = 0u; i < points.Count; ++i) {
            // An edge of more than two faces gives its point to the ring once, from the pair of halfedges that hold its outputs.
            if (points.Edge[i] != InvalidOffset && ctx.Src(job).IncomingEdge(points.Edge[i]) == InvalidOffset) continue;
            push(points.Vertex[i], h);
        }
        const uint out = ctx.SrcNext(job, h);
        const uint opposite = ctx.SrcOpposite(job, out);
        if (opposite != InvalidOffset && TopoEdgeBeveled(ctx, job, out) && segments > 1u) {
            bool same_way;
            const uint representative = TopoBevelRepresentative(ctx, job, out, same_way);
            // `out` leaves v, so v is the representative's from-end when `out` runs its way.
            const bool at_to = !same_way;
            const BevelHalfedgeOutputs o = TopoBevelOutputs(ctx, job, representative);
            const uint entry = ctx.HalfedgeEntry(job, representative);
            for (uint j = 0u; j + 1u < segments; ++j) {
                const uint index = same_way ? j : segments - 2u - j;
                push(TopoBevelKey(entry, o.ProfileIndex(at_to, index, segments)), h);
            }
        }
        if (opposite == InvalidOffset || opposite == start) break;
        h = opposite;
    }
    if (length > 1u && TopoBevelKeyEqual(first_vertex, previous_vertex)) --length;
    return {length, first_source};
}
