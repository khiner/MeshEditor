#ifndef MESHTOPOLOGYLINES_MSL
#define MESHTOPOLOGYLINES_MSL

// Edits the lines of a mesh without faces, whose core holds whole lines each as the two corners at its ends.
// Each output line is two adjacent output corners, the first at the line's end and the second at its start, as a line source writes them.
// A cut line emits its cut vertices and the pieces between them, and an extruded line emits its base and its top across the vertex copies.
// Any other line emits the one line it leaves: itself, its merged ends, or a dissolve chain's join, and a joining operator keeps one line per pair of ends.
#include "MeshTopologyContext.metal"
#include "MeshTopologySubdivide.metal"

// The one output line a source line leaves, between the source-local vertices whose outputs it joins.
struct TopoLine {
    uint From, To;
    bool Kept;
};

// A dissolve chain's far end: the first kept vertex past a removed one, and the representative corner of the chain's last line there.
struct TopoLineEnd {
    uint Vertex, Corner;
};

inline bool TopoLineJoins(MeshTopologyJob job) { return TopoLineCore(job) && TopologyJoinsLines(job.Op); }
inline uint2 TopoLineKey(TopoLine line) { return uint2(min(line.From, line.To), max(line.From, line.To)); }
inline uint TopoLineHash(uint2 key) { return TopoCellHash(int3(int(key.x), int(key.y), 0)); }

// Walks from the removed vertex `v`, entered through its corner `h`, along the lines of removed vertices to the first kept vertex.
// A chain that closes on itself has no end.
inline TopoLineEnd TopoLineChainEnd(TopoContext ctx, MeshTopologyJob job, uint v, uint h) {
    const auto src = ctx.Src(job);
    const auto corners = ctx.SrcCorners(job);
    for (uint step = 0u; step < job.SrcEdgeCount; ++step) {
        const uint2 fan = ctx.SrcFan(job, v);
        uint out = InvalidOffset;
        for (uint k = 0u; k < fan.y; ++k) {
            const uint corner = ctx.SrcFanCorner(job, fan.x + k);
            if (src.Edge(corner) != src.Edge(h)) out = corner;
        }
        if (out == InvalidOffset) break;
        h = src.Opposite(out);
        v = corners[h];
        if (!TopoVertexRemoved(ctx, job, v)) return {v, src.EdgeHalfedge(src.Edge(h))};
    }
    return {InvalidOffset, InvalidOffset};
}

// Whether a line outside the core joins the source-local vertices `a` and `b`.
inline bool TopoOuterLine(TopoContext ctx, MeshTopologyJob job, uint a, uint b) {
    const uint2 fan = ctx.SrcFan(job, a);
    for (uint k = 0u; k < fan.y; ++k) {
        const uint corner = ctx.SrcFanCorner(job, fan.x + k);
        if (ctx.SrcEdge(job, corner) == InvalidOffset && ctx.SrcCorners(job)[ctx.SrcOpposite(job, corner)] == b) return true;
    }
    return false;
}

// The one line a source line leaves, by its representative corner.
// A dissolve chain joins its kept ends at its lower end line, unless a line outside the core already joins them.
inline TopoLine TopoLineOutput(TopoContext ctx, MeshTopologyJob job, uint h) {
    const auto corners = ctx.SrcCorners(job);
    const uint pair = ctx.SrcPrev(job, h), a = corners[pair], b = corners[h];
    switch (job.Op) {
        case MeshTopologyOp::DeleteVertices: return {a, b, !ctx.SrcSelectedVertex(job, a) && !ctx.SrcSelectedVertex(job, b)};
        case MeshTopologyOp::DeleteEdges: return {a, b, !ctx.SrcSelectedEdge(job, ctx.SrcEdge(job, h))};
        case MeshTopologyOp::MergeAtTarget:
        case MeshTopologyOp::MergeByDistance:
        case MeshTopologyOp::MergeCollapse: {
            const uint from = ctx.VertexTargets(job)[a], to = ctx.VertexTargets(job)[b];
            return {from, to, from != to};
        }
        case MeshTopologyOp::DissolveVertices: {
            const bool removed_a = TopoVertexRemoved(ctx, job, a), removed_b = TopoVertexRemoved(ctx, job, b);
            if (removed_a == removed_b) return {a, b, !removed_a};
            const TopoLineEnd end = removed_a ? TopoLineChainEnd(ctx, job, a, pair) : TopoLineChainEnd(ctx, job, b, h);
            const uint from = removed_a ? end.Vertex : a, to = removed_a ? b : end.Vertex;
            const auto lines = ctx.SrcHalfedgeDomain(job);
            const bool lower = end.Corner != InvalidOffset && lines.Index(h) < lines.Index(end.Corner);
            return {from, to, lower && from != to && !TopoOuterLine(ctx, job, from, to)};
        }
        default: return {a, b, true};
    }
}

// Whether a kept line is the lowest source line that leaves a line between its ends, which the line key table records.
inline bool TopoLineFirst(TopoContext ctx, MeshTopologyJob job, TopoLine line, uint hi) {
    if (!TopoLineJoins(job)) return true;
    const uint2 key = TopoLineKey(line);
    device const uint *table = ctx.Table(job);
    uint slot = TopoLineHash(key) & job.TableMask;
    for (uint probe = 0u; probe <= job.TableMask; ++probe, slot = (slot + 1u) & job.TableMask) {
        const uint occupant = table[slot];
        if (occupant == hi) return true;
        if (occupant == InvalidOffset) return false;
        if (all(TopoLineKey(TopoLineOutput(ctx, job, ctx.SrcHalfedgeDomain(job).Handle(occupant))) == key)) return false;
    }
    return false;
}

// The output corners a source line emits at its representative corner.
inline uint TopoLineCorners(TopoContext ctx, MeshTopologyJob job, uint h) {
    if (job.Op == MeshTopologyOp::Subdivide && TopoEdgeCut(ctx, job, h)) return 2u * (TopoSubdivideCuts(job) + 1u);
    if (TopoHalfedgeMakesSide(ctx, job, h)) return 4u;
    const TopoLine line = TopoLineOutput(ctx, job, h);
    return line.Kept && TopoLineFirst(ctx, job, line, ctx.SrcHalfedgeDomain(job).Index(h)) ? 2u : 0u;
}

// Writes an output line from output vertex `from` to `to`, whose corners take their attributes from the source corners at its ends.
inline void TopoEmitLine(TopoContext ctx, MeshTopologyJob job, uint d, uint from, uint to, uint from_corner, uint to_corner, uint edge_source, bool selected) {
    ctx.WriteCorner(job, d, to, to_corner, to_corner, 0.f, edge_source, selected);
    ctx.WriteCorner(job, d + 1u, from, from_corner, from_corner, 0.f, edge_source, selected);
}

// Emits a source line's outputs at its representative corner.
inline void TopoScatterLine(TopoContext ctx, MeshTopologyJob job, uint h, uint entry) {
    const uint base = ctx.Counts(job, TopoCountCorners)[entry];
    if (ctx.Counts(job, TopoCountCorners)[entry + 1u] == base) return;
    device const uint *vertex_offsets = ctx.Counts(job, TopoCountVertices);
    const auto corners = ctx.SrcCorners(job);
    const uint pair = ctx.SrcPrev(job, h), from = corners[pair], to = corners[h];
    const bool selected = ctx.SrcSelectedEdge(job, ctx.SrcEdge(job, h));
    if (job.Op == MeshTopologyOp::Subdivide && TopoEdgeCut(ctx, job, h)) {
        const uint first = vertex_offsets[entry], cuts = vertex_offsets[entry + 1u] - first;
        uint previous = vertex_offsets[from];
        for (uint i = 0u; i <= cuts; ++i) {
            const uint next = i < cuts ? first + i : vertex_offsets[to];
            if (i < cuts) {
                ctx.WriteVertexMap(job, next, from, to, TopoCutParam(ctx, job, h, i, cuts));
                ctx.SelectDstVertex(job, next);
            }
            TopoEmitLine(ctx, job, base + 2u * i, previous, next, pair, h, h, selected);
            previous = next;
        }
        return;
    }
    // An extruded line's base stays in place and its top joins the endpoints' copies, which follow their own outputs.
    if (TopoHalfedgeMakesSide(ctx, job, h)) {
        TopoEmitLine(ctx, job, base, vertex_offsets[from], vertex_offsets[to], pair, h, h, false);
        TopoEmitLine(ctx, job, base + 2u, vertex_offsets[from] + 1u, vertex_offsets[to] + 1u, pair, h, h, true);
        return;
    }
    const TopoLine line = TopoLineOutput(ctx, job, h);
    TopoEmitLine(ctx, job, base, vertex_offsets[line.From], vertex_offsets[line.To], pair, h, h, selected);
}

#endif
