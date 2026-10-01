#ifndef MESHTOPOLOGYBEVEL_MSL
#define MESHTOPOLOGYBEVEL_MSL

// Bevels selected edges or vertices by width Param0 with Steps segments.
// A face corner at a beveled vertex moves onto offset points: a corner point when both of its edges are beveled, or a point along its other edge when one is.
// A corner with neither edge beveled uses the points its neighbors put on its edges.
// Each beveled edge becomes a strip of quads between its two sides, and each beveled vertex a polygon of the points around it.
#include "MeshTopologyContext.metal"
#include "MeshTopologyFaces.metal"


inline bool TopoBevelVertices(MeshTopologyJob job) { return job.Op == MeshTopologyOp::BevelVertices; }
inline uint TopoBevelSegments(MeshTopologyJob job) { return max(job.Steps, 1u); }
inline bool TopoEdgeBeveled(TopoContext ctx, MeshTopologyJob job, uint h) { return !TopoBevelVertices(job) && ctx.SrcSelectedEdge(job, ctx.SrcEdge(job, h)); }

// Whether a vertex is beveled: selected for a vertex bevel, or an end of a beveled edge.
inline bool TopoVertexBeveled(TopoContext ctx, MeshTopologyJob job, uint v) {
    if (TopoBevelVertices(job)) return ctx.SrcSelectedVertex(job, v);
    const uint2 fan = ctx.SrcFan(job, v);
    for (uint k = 0u; k < fan.y; ++k) {
        const uint h = ctx.SrcFanCorner(job,fan.x+k);
        if (TopoEdgeBeveled(ctx, job, h) || TopoEdgeBeveled(ctx, job, ctx.SrcNext(job, h))) return true;
    }
    return false;
}

// The corner at `h` (arriving at its vertex) has its arriving and leaving edges beveled as (in, out).
inline bool2 TopoCornerBeveled(TopoContext ctx, MeshTopologyJob job, uint h) {
    return bool2(TopoEdgeBeveled(ctx, job, h), TopoEdgeBeveled(ctx, job, ctx.SrcNext(job, h)));
}

// The distance along a corner's other edge where an offset line of its beveled edge meets it.
inline float TopoBevelSlide(float3 v, float3 along, float3 beveled, float width) {
    const float3 a = normalize(along - v), b = normalize(beveled - v);
    const float sine = length(cross(a, b));
    return width / max(sine, 1e-3f);
}

// The point a face corner contributes onto its edge `along` (a neighbor of `v`), sliding away from its beveled edge `beveled`.
inline float3 TopoBevelEdgePointFrom(TopoContext ctx, MeshTopologyJob job, uint v, uint along, uint beveled) {
    const float3 p = ctx.SrcPosition(job, v);
    const float3 a = ctx.SrcPosition(job, along);
    if (TopoBevelVertices(job)) return p + normalize(a - p) * job.Param0;
    return p + normalize(a - p) * TopoBevelSlide(p, a, ctx.SrcPosition(job, beveled), job.Param0);
}

// The point on unbeveled edge `rep` at its end `v`, from the corners on either side, or false when neither side contributes.
inline bool TopoBevelEdgePoint(TopoContext ctx, MeshTopologyJob job, uint rep, bool at_to, thread float3 &point) {
    const auto corners = ctx.SrcCorners(job);
    const uint opposite = ctx.SrcOpposite(job, rep);
    const uint v = at_to ? corners[rep] : corners[ctx.SrcPrev(job, rep)];
    const uint other = at_to ? corners[ctx.SrcPrev(job, rep)] : corners[rep];
    if (TopoBevelVertices(job)) {
        if (!ctx.SrcSelectedVertex(job, v)) return false;
        point = TopoBevelEdgePointFrom(ctx, job, v, other, other);
        return true;
    }
    // The corner in each face whose other edge is beveled slides its point along this edge.
    float best = -1.f;
    const auto consider = [&](uint corner_halfedge, bool arriving) {
        // `arriving`: the corner's arriving edge is this edge, so its leaving edge is the beveled candidate.
        const uint candidate = arriving ? ctx.SrcNext(job, corner_halfedge) : corner_halfedge;
        if (!TopoEdgeBeveled(ctx, job, candidate)) return;
        const uint far = arriving ? corners[candidate] : corners[ctx.SrcPrev(job, candidate)];
        const float3 p = TopoBevelEdgePointFrom(ctx, job, v, other, far);
        const float d = distance(p, ctx.SrcPosition(job, v));
        if (d > best) {
            best = d;
            point = p;
        }
    };
    if (at_to) {
        consider(rep, true);
        if (opposite != InvalidOffset) consider(ctx.SrcPrev(job, opposite), false);
    } else {
        consider(ctx.SrcPrev(job, rep), false);
        if (opposite != InvalidOffset) consider(opposite, true);
    }
    return best >= 0.f;
}

inline float3 TopoBevelCornerPoint(TopoContext ctx, MeshTopologyJob job, uint h) {
    const auto corners = ctx.SrcCorners(job);
    const uint prev = ctx.SrcPrev(job, h), next = ctx.SrcNext(job, h);
    return ctx.SrcPosition(job, corners[h]) + TopoInsetCorner(ctx.SrcPosition(job, corners[prev]), ctx.SrcPosition(job, corners[h]), ctx.SrcPosition(job, corners[next]), normalize(float3(ctx.SrcFaceNormals(job)[ctx.SrcFaceOf(job, h)])), job.Param0, 0.f, true);
}

// The vertices a halfedge entry produces: its edge's points at both ends when it is an unbeveled representative,
// its corner point when both corner edges are beveled, and its profile points when it is a beveled representative.
struct BevelHalfedgeOutputs {
    bool EndFrom, EndTo, Corner;
    uint Profiles; // 2 * (segments - 1) for a beveled representative
    uint Count() const { return (EndFrom ? 1u : 0u) + (EndTo ? 1u : 0u) + (Corner ? 1u : 0u) + Profiles; }
    uint EndIndex(bool at_to) const { return at_to ? (EndFrom ? 1u : 0u) : 0u; }
    uint CornerIndex() const { return (EndFrom ? 1u : 0u) + (EndTo ? 1u : 0u); }
    uint ProfileIndex(bool at_to, uint j, uint segments) const { return CornerIndex() + (Corner ? 1u : 0u) + (at_to ? segments - 1u : 0u) + j; }
};

inline BevelHalfedgeOutputs TopoBevelOutputs(TopoContext ctx, MeshTopologyJob job, uint h) {
    BevelHalfedgeOutputs o{false, false, false, 0u};
    const bool rep = ctx.SrcEdgeFirst(job, h);
    const bool beveled = TopoEdgeBeveled(ctx, job, h);
    float3 point;
    if (rep && !beveled) {
        o.EndFrom = TopoBevelEdgePoint(ctx, job, h, false, point);
        o.EndTo = TopoBevelEdgePoint(ctx, job, h, true, point);
    }
    const bool2 both = TopoCornerBeveled(ctx, job, h);
    o.Corner = both.x && both.y;
    if (rep && beveled) o.Profiles = 2u * (TopoBevelSegments(job) - 1u);
    return o;
}

// Count passes compare these keys before the scan.
// Emission resolves them to output ordinals.
// Keeping entry and local index separate imposes no per-entry vertex limit.
inline uint2 TopoBevelKey(uint entry, uint local) { return uint2(entry, local); }
inline bool TopoBevelKeyValid(uint2 key) { return key.x != InvalidOffset; }
inline bool TopoBevelKeyEqual(uint2 a, uint2 b) { return all(a == b); }
inline uint TopoBevelVertexOf(TopoContext ctx, MeshTopologyJob job, uint2 key) {
    return TopoBevelKeyValid(key) ? ctx.Counts(job, TopoCountVertices)[key.x] + key.y : InvalidOffset;
}

inline uint TopoBevelVertexRingIndex(TopoContext ctx, MeshTopologyJob job, uint entry, uint2 boundary,
                                   uint index, uint length, uint ring) {
    return ring == 0u ? TopoBevelVertexOf(ctx, job, boundary) :
        ctx.Counts(job, TopoCountVertices)[entry] + (ring - 1u) * length + index;
}

// The representative halfedge of `h`'s edge, and whether `h` is it.
inline uint TopoBevelRepresentative(TopoContext ctx, MeshTopologyJob job, uint h, thread bool &is_rep) {
    is_rep = ctx.SrcEdgeFirst(job, h);
    return is_rep ? h : ctx.SrcOpposite(job, h);
}

// The key of the point on halfedge `h`'s edge at its end `at_to`, or InvalidOffset when the edge has none there.
inline uint2 TopoBevelEdgeVertex(TopoContext ctx, MeshTopologyJob job, uint h, bool at_to) {
    bool rep;
    const uint representative = TopoBevelRepresentative(ctx, job, h, rep);
    if (representative == InvalidOffset) return uint2(InvalidOffset);
    // The opposite halfedge runs the other way, so its ends swap.
    const bool end = rep ? at_to : !at_to;
    const BevelHalfedgeOutputs o = TopoBevelOutputs(ctx, job, representative);
    if (end ? !o.EndTo : !o.EndFrom) return uint2(InvalidOffset);
    return TopoBevelKey(ctx.HalfedgeEntry(job, representative), o.EndIndex(end));
}

inline float3 TopoBevelVertexRingPosition(TopoContext ctx, MeshTopologyJob job, uint v, uint2 boundary,
                                        uint source, uint ring) {
    const auto corners = ctx.SrcCorners(job);
    const uint along = TopoBevelKeyEqual(boundary, TopoBevelEdgeVertex(ctx, job, source, true)) ?
        corners[ctx.SrcPrev(job, source)] : corners[ctx.SrcNext(job, source)];
    const float3 center = ctx.SrcPosition(job, v);
    const float3 radial = normalize(ctx.SrcPosition(job, along) - center);
    const float t = float(ring) / float(TopoBevelSegments(job));
    const float3 normal = normalize(float3(ctx.SrcVertexNormals(job)[v]));
    const float convex = max(-dot(radial, normal), 0.f);
    const float remaining = 1.f - t;
    return center + job.Param0 * (radial * remaining * remaining - normal * (0.5f * convex * t * t));
}

inline uint2 TopoBevelCornerVertex(TopoContext ctx, MeshTopologyJob job, uint h) {
    return TopoBevelKey(ctx.HalfedgeEntry(job, h), TopoBevelOutputs(ctx, job, h).CornerIndex());
}

// A corner's replacement points: up to two point keys with the corner as their attribute source.
struct BevelCornerPoints {
    uint Count;
    uint2 Vertex[2];
};

inline BevelCornerPoints TopoBevelCornerPoints(TopoContext ctx, MeshTopologyJob job, uint h) {
    BevelCornerPoints result{0u, {uint2(InvalidOffset), uint2(InvalidOffset)}};
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
    // Each unbeveled edge at the corner contributes the point on it.
    const uint next = ctx.SrcNext(job, h);
    const uint2 on_in = both.x ? uint2(InvalidOffset) : TopoBevelEdgeVertex(ctx, job, h, true);
    const uint2 on_out = both.y ? uint2(InvalidOffset) : TopoBevelEdgeVertex(ctx, job, next, false);
    if (TopoBevelKeyValid(on_in)) result.Vertex[result.Count++] = on_in;
    if (TopoBevelKeyValid(on_out) && !TopoBevelKeyEqual(on_out, on_in)) result.Vertex[result.Count++] = on_out;
    if (result.Count == 0u && !both.x && !both.y) result.Vertex[result.Count++] = TopoBevelKey(ctx.VertexEntry(v), 0u);
    return result;
}

// The position of the side point of a beveled edge in its own halfedge's face at end `at_to`.
inline float3 TopoBevelSidePosition(TopoContext ctx, MeshTopologyJob job, uint edge_halfedge, bool at_to) {
    const uint corner = at_to ? edge_halfedge : ctx.SrcPrev(job, edge_halfedge);
    const bool2 both = TopoCornerBeveled(ctx, job, corner);
    if (both.x && both.y) return TopoBevelCornerPoint(ctx, job, corner);
    // The other edge at the corner carries the side point: the corner's leaving edge at the to-end, its arriving edge at the from-end.
    const uint other = at_to ? ctx.SrcNext(job, corner) : corner;
    bool other_rep;
    const uint representative = TopoBevelRepresentative(ctx, job, other, other_rep);
    float3 point = ctx.SrcPosition(job, ctx.SrcCorners(job)[corner]);
    if (representative != InvalidOffset) TopoBevelEdgePoint(ctx, job, representative, other_rep ? at_to : !at_to, point);
    return point;
}

// The key of the side point of a beveled edge in its own halfedge's face at end `at_to`: the face's corner point there.
inline uint2 TopoBevelSideVertex(TopoContext ctx, MeshTopologyJob job, uint edge_halfedge, bool at_to) {
    // The corner of this halfedge's face at the end: the halfedge itself arrives at its to-vertex, and its predecessor at its from-vertex.
    const uint corner = at_to ? edge_halfedge : ctx.SrcPrev(job, edge_halfedge);
    const BevelCornerPoints points = TopoBevelCornerPoints(ctx, job, corner);
    // The point nearest this edge: a corner whose other edge is unbeveled slid onto it, or the corner point.
    return at_to ? points.Vertex[0] : points.Vertex[points.Count - 1u];
}

// Walks the corners around vertex `v` in rotation order from an open side.
// The count and scatter passes use the same walk.
// Scatter writes directly to its counted output.
struct BevelRingInfo { uint Length, FirstSource; };
template<typename Emit>
inline BevelRingInfo TopoBevelRing(TopoContext ctx, MeshTopologyJob job, uint v, Emit emit) {
    const uint2 fan = ctx.SrcFan(job, v);
    if (fan.y == 0u) return {0u, InvalidOffset};
    // Start from the corner whose arriving edge is open, or the lowest corner on a closed fan.
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
        for (uint i = 0u; i < points.Count; ++i) push(points.Vertex[i], h);
        const uint out = ctx.SrcNext(job, h);
        const uint opposite = ctx.SrcOpposite(job, out);
        // Crossing a beveled edge passes its profile points at this end, from this face's side to the next face's.
        if (opposite != InvalidOffset && TopoEdgeBeveled(ctx, job, out) && segments > 1u) {
            bool rep;
            const uint representative = TopoBevelRepresentative(ctx, job, out, rep);
            // `out` leaves v, so v is the representative's from-end when `out` is the representative.
            const bool at_to = !rep;
            const BevelHalfedgeOutputs o = TopoBevelOutputs(ctx, job, representative);
            const uint entry = ctx.HalfedgeEntry(job, representative);
            // Profiles run from the representative's face side to its opposite's side.
            for (uint j = 0u; j + 1u < segments; ++j) {
                const uint index = rep ? j : segments - 2u - j;
                push(TopoBevelKey(entry, o.ProfileIndex(at_to, index, segments)), h);
            }
        }
        if (opposite == InvalidOffset || opposite == start) break;
        h = opposite;
    }
    // A closed ring drops a repeated first point.
    if (length > 1u && TopoBevelKeyEqual(first_vertex, previous_vertex)) --length;
    return {length, first_source};
}

#endif
