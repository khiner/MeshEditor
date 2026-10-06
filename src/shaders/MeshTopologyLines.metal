#ifndef MESHTOPOLOGYLINES_MSL
#define MESHTOPOLOGYLINES_MSL

// Edits explicit wires and the loose edges left by collapsed faces.
// Each output line is two adjacent output corners, the first at the line's end and the second at its start, as a line source writes them.
// A cut line emits its cut vertices and the pieces between them.
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

inline bool TopoLineJoins(MeshTopologyJob job) { return TopologyJoinsLines(job.Op); }
inline uint2 TopoLineKey(TopoLine line) { return uint2(min(line.From, line.To), max(line.From, line.To)); }
inline uint TopoLineHash(uint2 key) { return TopoCellHash(int3(int(key.x), int(key.y), 0)); }

// Record the preceding kept corner once per dissolved boundary. A two-corner
// boundary leaves one loose edge; longer boundaries continue to own face edges.
inline void TopoDissolveBoundary(TopoContext ctx, MeshTopologyJob job, uint f) {
    const bool own=TopoDissolveOwnLoop(ctx,job,f);
    if (!own && ctx.FaceLabels(job)[f]!=f) return;
    const uint2 range=ctx.SrcFaceRange(job,f);
    const uint start=own ? range.x : ctx.RegionStart(job)[f];
    const uint count=own ? TopoMappedLoopLength(ctx,job,f) : ctx.WalkLength(job)[f];
    if (count<2u) return;
    uint h=start,first=InvalidOffset,previous=InvalidOffset;
    do {
        if (!TopoVertexRemoved(ctx,job,ctx.SrcCorners(job)[h])) {
            if (first==InvalidOffset) first=h;
            ctx.DissolvePrevious(job)[h]=previous;
            previous=h;
            if (count>=3u) ctx.FlagHalfedges(job)[h]|=TopoSurfaceEdge;
        }
        h=own ? ctx.SrcNext(job,h) : TopoNextBoundary(ctx,job,h);
    } while (h!=start);
    ctx.DissolvePrevious(job)[first]=previous;
}

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
            if (src.Edge(corner) != src.Edge(h) && !(ctx.FlagHalfedges(job)[corner]&TopoWireRemoved)) out = corner;
        }
        if (out == InvalidOffset) break;
        h = src.Opposite(out);
        v = corners[h];
        if (!TopoVertexRemoved(ctx, job, v)) return {v, src.EdgeHalfedge(src.Edge(h))};
    }
    return {InvalidOffset, InvalidOffset};
}

// Only unchanged edges outside the core need a fan query. Inside the core,
// the shared key table prefers surviving face edges over coincident wires.
inline bool TopoOuterLine(TopoContext ctx, MeshTopologyJob job, uint a, uint b) {
    const auto src=ctx.Src(job);
    for (const auto item:src.Fan(ctx.SrcVertexDomain(job).Handle(a))) {
        const uint h=item.x;
        if (item.y==InvalidOffset) {
            if (ctx.SrcEdge(job,h)==InvalidOffset && ctx.SrcCorners(job)[src.Opposite(h)]==b) return true;
        } else if (ctx.SrcFaceDomain(job).Index(item.y)==InvalidOffset) {
            if (ctx.SrcCorners(job)[src.Next(h,item.y)]==b || ctx.SrcCorners(job)[ctx.SrcPrev(job,h)]==b) return true;
        }
    }
    return false;
}

// The one line a source line leaves, by its representative corner.
// A dissolve chain joins its kept ends at its lower end line, unless a line outside the core already joins them.
inline TopoLine TopoLineOutput(TopoContext ctx, MeshTopologyJob job, uint h) {
    const auto corners = ctx.SrcCorners(job);
    const uint pair = ctx.SrcPrev(job, h), a = corners[pair], b = corners[h];
    if (TopologyIsDissolve(job.Op) && TopoLineJoins(job) && ctx.Src(job).HalfedgeFace(h)!=InvalidOffset) {
        const uint previous=ctx.DissolvePrevious(job)[h];
        if (previous==InvalidOffset) return {a,b,false};
        const uint from=corners[previous];
        return {from,b,(ctx.FlagHalfedges(job)[h]&TopoSurfaceEdge) || !TopoOuterLine(ctx,job,from,b)};
    }
    switch (job.Op) {
        case MeshTopologyOp::KeepSelectedFaces: return {a,b,ctx.SrcSelectedEdge(job,ctx.SrcEdge(job,h))};
        case MeshTopologyOp::MergeAtTarget:
        case MeshTopologyOp::MergeByDistance:
        case MeshTopologyOp::MergeCollapse:
        case MeshTopologyOp::DissolveDegenerate:
        case MeshTopologyOp::Decimate: {
            const uint from = ctx.VertexTargets(job)[a], to = ctx.VertexTargets(job)[b];
            return {from, to, from != to};
        }
        case MeshTopologyOp::DissolveVertices:
        case MeshTopologyOp::DissolveEdges:
        case MeshTopologyOp::DissolveLimited: {
            if (ctx.FlagHalfedges(job)[h]&TopoWireRemoved) return {a,b,false};
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

// A surviving face owns its mapped edge in preference to any coincident wire.
inline bool TopoLineSurface(TopoContext ctx, MeshTopologyJob job, uint h) {
    return (TopologyIsMerge(job.Op) || (TopologyIsDissolve(job.Op) && TopoLineJoins(job))) && (ctx.FlagHalfedges(job)[h]&TopoSurfaceEdge)!=0u;
}

// Reuse an existing edge before a chain or weld would create another at its endpoints.
inline uint TopoLinePriority(TopoContext ctx, MeshTopologyJob job, uint h, TopoLine line) {
    if (TopoLineSurface(ctx,job,h)) return 0u;
    const auto corners=ctx.SrcCorners(job);
    return all(TopoLineKey(line)==uint2(min(corners[h],corners[ctx.SrcPrev(job,h)]),max(corners[h],corners[ctx.SrcPrev(job,h)]))) ? 1u : 2u;
}

// The key table chooses a surviving face corner, or the lowest loose-edge representative.
inline uint TopoLineRepresentative(TopoContext ctx, MeshTopologyJob job, TopoLine line) {
    const uint2 key = TopoLineKey(line);
    device const uint *table = ctx.Table(job);
    uint slot = TopoLineHash(key) & job.TableMask;
    for (uint probe = 0u; probe <= job.TableMask; ++probe, slot = (slot + 1u) & job.TableMask) {
        const uint occupant = table[slot];
        if (occupant == InvalidOffset) return InvalidOffset;
        if (all(TopoLineKey(TopoLineOutput(ctx, job, ctx.SrcHalfedgeDomain(job).Handle(occupant))) == key)) return occupant;
    }
    return InvalidOffset;
}

inline bool TopoLineFirst(TopoContext ctx, MeshTopologyJob job, TopoLine line, uint hi) {
    if (!TopoLineJoins(job)) return true;
    return TopoLineRepresentative(ctx,job,line)==hi && !TopoLineSurface(ctx,job,ctx.SrcHalfedgeDomain(job).Handle(hi));
}

// The output corners a source line emits at its representative corner.
inline uint TopoLineCorners(TopoContext ctx, MeshTopologyJob job, uint h) {
    if (job.Op == MeshTopologyOp::Subdivide && TopoEdgeCut(ctx, job, h)) return 2u * (TopoSubdivideCuts(job) + 1u);
    const TopoLine line = TopoLineOutput(ctx, job, h);
    return line.Kept && TopoLineFirst(ctx, job, line, ctx.SrcHalfedgeDomain(job).Index(h)) ? 2u : 0u;
}

// Writes an output line from output vertex `from` to `to`, whose corners take their attributes from the source corners at its ends.
inline void TopoEmitLine(TopoContext ctx, MeshTopologyJob job, uint d, uint from, uint to, uint from_corner, uint to_corner, uint edge_source, bool selected) {
    ctx.DstHalfedgeFaces(job)[d]=ctx.DstHalfedgeFaces(job)[d+1u]=InvalidOffset;
    device uint *opposites=BindlessBufferMutable(uint,ctx.B.Buffer,job.DstConnectivity.Opposites.Slot)+job.DstCornerOffset;
    opposites[d]=job.DstCornerOffset+d+1u; opposites[d+1u]=job.DstCornerOffset+d;
    ctx.WriteCorner(job, d, to, to_corner, to_corner, 0.f, edge_source, selected);
    ctx.WriteCorner(job, d + 1u, from, from_corner, from_corner, 0.f, edge_source, selected);
}

// Emits a source line's outputs at its representative corner.
inline void TopoScatterLine(TopoContext ctx, MeshTopologyJob job, uint h, uint entry) {
    if (ctx.Counts(job, TopoCountWireCorners)[entry + 1u] == ctx.Counts(job, TopoCountWireCorners)[entry]) return;
    const uint base = ctx.WireCornerOffset(job, entry);
    device const uint *vertex_offsets = ctx.Counts(job, TopoCountVertices);
    const auto corners = ctx.SrcCorners(job);
    const uint pair = TopologyIsDissolve(job.Op) && ctx.Src(job).HalfedgeFace(h)!=InvalidOffset ? ctx.DissolvePrevious(job)[h] : ctx.SrcPrev(job,h);
    const uint from=corners[pair],to=corners[h];
    const bool selected = ctx.SrcSelectedEdge(job, ctx.SrcEdge(job, h));
    if (TopologyCopiesSelection(job.Op)) {
        const bool own = ctx.Src(job).HalfedgeFace(h) == InvalidOffset && (job.Op != MeshTopologyOp::SplitGeometry || !selected);
        if (own) TopoEmitLine(ctx, job, base, vertex_offsets[from], vertex_offsets[to], pair, h, h, false);
        if (ctx.Counts(job, TopoCountWireCorners)[entry + 1u] - ctx.Counts(job, TopoCountWireCorners)[entry] > 2u * uint(own))
            TopoEmitLine(ctx, job, base + 2u * uint(own), vertex_offsets[from] + TopoVertexCopies(ctx, job, from),
                vertex_offsets[to] + TopoVertexCopies(ctx, job, to), pair, h, h, true);
        return;
    }
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
    const TopoLine line = TopoLineOutput(ctx, job, h);
    TopoEmitLine(ctx, job, base, vertex_offsets[line.From], vertex_offsets[line.To], pair, h, h, selected && !TopologyExtrudesSides(job.Op));
}

#endif
