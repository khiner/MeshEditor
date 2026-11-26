#include "BlockScan.metal"
#include "ConnectivityRead.metal"
#include "ListRank.metal"
#include "gpu/SelectedChainsJob.h"

// The chains of a record's selected edges.
struct ChainContext {
    device const BindlessSet &B;
    constant SelectedChainsJob &Pc;
    ConnectivityView View;
    device const Vertex *Vertices;
    JobRuns Runs;

    ChainContext(device const BindlessSet &b, constant JobTablePushConstants &table, constant SelectedChainsJob &pc)
        : B(b), Pc(pc), View{b, pc.Connectivity}, Vertices(BindlessBuffer(Vertex, b.VertexBuffer, pc.VertexSlot)), Runs(ScratchOf(b, table)) {}
    device packed_uint2 *PairRun(uint offset) const { return reinterpret_cast<device packed_uint2 *>(Runs.Run(offset)); }
    uint Handle(uint i) const { return BlockHandle(i, B, Pc.Blocks); }
    float3 Position(uint v) const { return float3(Vertices[View.VertexAddress(v)].Position); }
    uint2 Span(uint o) const {
        const uint2 ends = View.EdgeEnds(o / 2u);
        return (o & 1u) ? ends.yx : ends;
    }
    bool Listed(uint edge) const { return SortedContains(Runs.Run(Pc.Edges), Pc.EdgeCount, edge); }
    bool IsCandidate(uint o) const { return o != InvalidOffset && Runs.Run(Pc.Candidate)[o] != 0u; }
    // The halfedge supplying an oriented edge's corner attributes, which is its edge's first halfedge.
    uint Source(uint o) const {
        const uint e = o / 2u;
        return Pc.BoundaryOnly && View.EdgeSides(e) > 1u ? InvalidOffset : View.EdgeHalfedge(e);
    }
    static bool Before(uint2 key, uint2 other) { return key.x < other.x || (key.x == other.x && key.y < other.y); }
};
// Declares a job table kernel's context, its job `pc` and the thread's index `index` within the job.
#define CHAIN_THREAD(index) JOB_THREAD(pc, index); const ChainContext ctx{b, table, pc}

// Each listed edge with two distinct ends makes both its orientations candidates and counts itself at each end.
kernel void SelectedChainDegrees(JOB_TABLE_ARGS(SelectedChainsJob)) {
    CHAIN_THREAD(i);
    const uint e = ctx.Handle(i);
    if (e == InvalidOffset) return;
    if (!ctx.Listed(e)) return;
    if (pc.BoundaryOnly && ctx.View.EdgeSides(e) > 1u) return;
    const uint2 ends = ctx.View.EdgeEnds(e);
    if (ends.x == ends.y) return;
    ctx.Runs.Run(pc.Candidate)[2u * e] = ctx.Runs.Run(pc.Candidate)[2u * e + 1u] = 1u;
    for (uint side = 0u; side < 2u; ++side) {
        const uint v = side ? ends.y : ends.x;
        atomic_fetch_add_explicit(AtomicWord(ctx.Runs.Run(pc.Degree) + v), 1u, memory_order_relaxed);
        atomic_fetch_min_explicit(AtomicWord(ctx.Runs.Run(pc.Lowest) + v), e, memory_order_relaxed);
        atomic_fetch_max_explicit(AtomicWord(ctx.Runs.Run(pc.Highest) + v), e, memory_order_relaxed);
    }
}

kernel void SelectedChainFaceEdges(JOB_TABLE_ARGS(SelectedChainsJob)) {
    CHAIN_THREAD(i);
    if (i >= pc.ListedFaces) return;
    const uint2 range = ctx.View.FaceHalfedges(ctx.Runs.Run(pc.Faces)[i]);
    for (uint h = range.x; h < range.y; ++h) atomic_fetch_add_explicit(AtomicWord(ctx.Runs.Run(pc.Counts) + ctx.View.Edge(h)), 1u, memory_order_relaxed);
}
kernel void SelectedChainEdgeFacts(JOB_TABLE_ARGS(SelectedChainsJob)) {
    CHAIN_THREAD(i);
    if (i >= pc.EdgeCount) return;
    const uint e = ctx.Runs.Run(pc.Edges)[i];
    const uint2 ends = ctx.View.EdgeEnds(e);
    reinterpret_cast<device EdgeFacts *>(ctx.Runs.Run(pc.EdgeInfo))[i] = {ends.x, ends.y, ctx.Source(2u * e), ctx.View.HalfedgeFace(ctx.View.EdgeHalfedge(e)), ctx.Runs.Run(pc.Counts)[e]};
}

// Each orientation of a candidate links to the other selected edge at its end, oriented away from it, when exactly two meet there, and flags an end with more.
kernel void SelectedChainLink(JOB_TABLE_ARGS(SelectedChainsJob)) {
    CHAIN_THREAD(i);
    const uint e = ctx.Handle(i);
    if (e == InvalidOffset || !ctx.IsCandidate(2u * e)) return;
    for (uint d = 0u; d < 2u; ++d) {
        const uint o = 2u * e + d;
        const uint2 span = ctx.Span(o);
        const uint degree = ctx.Runs.Run(pc.Degree)[span.y];
        uint next = InvalidOffset;
        if (degree > 2u) {
            ctx.Runs.Run(pc.Junctions)[0] = 1u;
        } else if (degree == 2u) {
            const uint other = ctx.Runs.Run(pc.Lowest)[span.y] == e ? ctx.Runs.Run(pc.Highest)[span.y] : ctx.Runs.Run(pc.Lowest)[span.y];
            next = ctx.View.EdgeEnds(other).x == span.y ? 2u * other : 2u * other + 1u;
            ctx.Runs.Run(pc.Previous)[next] = o;
        }
        ctx.Runs.Run(pc.Next)[o] = next;
        ctx.Runs.Run(pc.Jumps)[o] = next;
        ctx.PairRun(pc.Key)[o] = packed_uint2(span);
        ctx.Runs.Run(pc.KeyIndex)[o] = o;
        ctx.PairRun(pc.Reverse)[o] = packed_uint2(span.yx);
        ctx.Runs.Run(pc.Tails)[o] = next == InvalidOffset ? o : InvalidOffset;
    }
}
kernel void SelectedChainHeads(JOB_TABLE_ARGS(SelectedChainsJob)) {
    CHAIN_THREAD(i);
    const uint e = ctx.Handle(i);
    if (e == InvalidOffset || !ctx.IsCandidate(2u * e)) return;
    for (uint d = 0u; d < 2u; ++d) {
        const uint o = 2u * e + d, previous = ctx.Runs.Run(pc.Previous)[o];
        ctx.Runs.Run(pc.PreviousJumps)[o] = previous;
        ctx.Runs.Run(pc.Heads)[o] = previous == InvalidOffset ? o : InvalidOffset;
    }
}

// Each round takes the lower keys a jump away, learns the tail a forward jump knows and the head a backward jump knows, and doubles both jumps.
kernel void SelectedChainLabel(JOB_TABLE_ARGS(SelectedChainsJob)) {
    CHAIN_THREAD(i);
    const uint e = ctx.Handle(i);
    if (e == InvalidOffset || !ctx.IsCandidate(2u * e)) return;
    const uint bound = 2u * pc.EdgeBound, in = (table.Parameter & 1u) * bound, out = bound - in;
    device packed_uint2 *keys = ctx.PairRun(pc.Key), *reverses = ctx.PairRun(pc.Reverse);
    for (uint d = 0u; d < 2u; ++d) {
        const uint o = 2u * e + d;
        const uint j = ctx.Runs.Run(pc.Jumps)[in + o], p = ctx.Runs.Run(pc.PreviousJumps)[in + o];
        uint2 key = uint2(keys[in + o]), reverse = uint2(reverses[in + o]);
        uint index = ctx.Runs.Run(pc.KeyIndex)[in + o], tail = ctx.Runs.Run(pc.Tails)[in + o], head = ctx.Runs.Run(pc.Heads)[in + o];
        if (j != InvalidOffset) {
            const uint2 other = uint2(keys[in + j]), other_reverse = uint2(reverses[in + j]);
            if (ChainContext::Before(other, key)) {
                key = other;
                index = ctx.Runs.Run(pc.KeyIndex)[in + j];
            }
            if (ChainContext::Before(other_reverse, reverse)) reverse = other_reverse;
            if (tail == InvalidOffset) tail = ctx.Runs.Run(pc.Tails)[in + j];
        }
        if (p != InvalidOffset && head == InvalidOffset) head = ctx.Runs.Run(pc.Heads)[in + p];
        ctx.Runs.Run(pc.Jumps)[out + o] = j == InvalidOffset ? j : ctx.Runs.Run(pc.Jumps)[in + j];
        ctx.Runs.Run(pc.PreviousJumps)[out + o] = p == InvalidOffset ? p : ctx.Runs.Run(pc.PreviousJumps)[in + p];
        keys[out + o] = packed_uint2(key);
        ctx.Runs.Run(pc.KeyIndex)[out + o] = index;
        reverses[out + o] = packed_uint2(reverse);
        ctx.Runs.Run(pc.Tails)[out + o] = tail;
        ctx.Runs.Run(pc.Heads)[out + o] = head;
    }
}

// A chain whose forward jumps never ran out is closed, and its representative is its lowest keyed oriented edge, else its head.
// Parameter holds the final label buffers' parity.
kernel void SelectedChainRankInit(JOB_TABLE_ARGS(SelectedChainsJob)) {
    CHAIN_THREAD(i);
    const uint e = ctx.Handle(i);
    if (e == InvalidOffset || !ctx.IsCandidate(2u * e)) return;
    const uint in = table.Parameter * 2u * pc.EdgeBound;
    for (uint d = 0u; d < 2u; ++d) {
        const uint o = 2u * e + d, next = ctx.Runs.Run(pc.Next)[o];
        const bool closed = ctx.Runs.Run(pc.Jumps)[in + o] != InvalidOffset;
        ctx.Runs.Run(pc.Closed)[o] = closed ? 1u : 0u;
        ctx.Runs.Run(pc.Distances)[o] = 1u;
        ctx.Runs.Run(pc.Successors)[o] = closed && next == ctx.Runs.Run(pc.KeyIndex)[in + o] ? InvalidOffset : next;
    }
}
kernel void SelectedChainRank(JOB_TABLE_ARGS(SelectedChainsJob)) {
    CHAIN_THREAD(i);
    const uint e = ctx.Handle(i);
    if (e == InvalidOffset || !ctx.IsCandidate(2u * e)) return;
    for (uint d = 0u; d < 2u; ++d) RankRound(ctx.Runs.Run(pc.Distances), ctx.Runs.Run(pc.Successors), 2u * pc.EdgeBound, table.Parameter & 1u, 2u * e + d);
}

// Each representative whose direction is the lower keyed one counts its chain's vertices and itself for the scans, with a zero terminator.
// Parameter holds the final buffers' parity, which the terminator's thread writes, one thread per oriented edge and one for the terminator.
kernel void SelectedChainCount(JOB_TABLE_ARGS(SelectedChainsJob)) {
    CHAIN_THREAD(i);
    const uint bound = 2u * pc.EdgeBound;
    if (i > bound) return;
    device uint *lengths = ctx.Runs.Run(pc.Lengths), *flags = ctx.Runs.Run(pc.Flags);
    lengths[i] = flags[i] = 0u;
    if (i == bound) ctx.Runs.Run(pc.Parity)[0] = table.Parameter;
    if (i == bound || !ctx.IsCandidate(i)) return;
    const uint converged = table.Parameter * bound;
    const bool closed = ctx.Runs.Run(pc.Closed)[i] != 0u;
    bool kept = false;
    if (closed) {
        if (ctx.Runs.Run(pc.KeyIndex)[converged + i] != i) return;
        kept = ChainContext::Before(uint2(ctx.PairRun(pc.Key)[converged + i]), uint2(ctx.PairRun(pc.Reverse)[converged + i]));
    } else {
        if (ctx.Runs.Run(pc.Previous)[i] != InvalidOffset) return;
        const uint2 tail = ctx.Span(ctx.Runs.Run(pc.Tails)[converged + i]);
        kept = ChainContext::Before(ctx.Span(i), tail.yx);
    }
    if (!kept) return;
    lengths[i] = ctx.Runs.Run(pc.Distances)[converged + i] + (closed ? 0u : 1u);
    flags[i] = 1u;
}

// Each member of a kept chain writes its origin and the source of the oriented edge before it at its place, and an open chain's tail adds its end.
kernel void SelectedChainOutput(JOB_TABLE_ARGS(SelectedChainsJob)) {
    CHAIN_THREAD(i);
    const uint e = ctx.Handle(i);
    if (e == InvalidOffset || !ctx.IsCandidate(2u * e)) return;
    const uint converged = ctx.Runs.Run(pc.Parity)[0] * 2u * pc.EdgeBound;
    device const uint *lengths = ctx.Runs.Run(pc.Lengths), *flags = ctx.Runs.Run(pc.Flags);
    for (uint d = 0u; d < 2u; ++d) {
        const uint o = 2u * e + d;
        const bool closed = ctx.Runs.Run(pc.Closed)[o] != 0u;
        const uint rep = closed ? ctx.Runs.Run(pc.KeyIndex)[converged + o] : ctx.Runs.Run(pc.Heads)[converged + o];
        if (flags[rep + 1u] == flags[rep]) continue;
        const uint offset = lengths[rep], edges = ctx.Runs.Run(pc.Distances)[converged + rep];
        if (rep == o) {
            device uint *record = ctx.Runs.Run(pc.Records) + 3u * flags[rep];
            record[0] = offset;
            record[1] = lengths[rep + 1u] - offset;
            record[2] = closed ? 1u : 0u;
        }
        const uint previous = ctx.Runs.Run(pc.Previous)[o];
        const uint2 span = ctx.Span(o);
        device uint *pair = ctx.Runs.Run(pc.Output) + 2u * (offset + edges - ctx.Runs.Run(pc.Distances)[converged + o]);
        pair[0] = span.x;
        pair[1] = previous == InvalidOffset ? InvalidOffset : ctx.Source(previous);
        if (!closed && ctx.Runs.Run(pc.Next)[o] == InvalidOffset) {
            device uint *last = ctx.Runs.Run(pc.Output) + 2u * (offset + edges);
            last[0] = span.y;
            last[1] = ctx.Source(o);
        }
    }
}

// Each chain records the winding of the face halfedges supplying its corners and whether a face spans exactly its vertices.
kernel void SelectedChainSummaries(JOB_TABLE_ARGS(SelectedChainsJob)) {
    CHAIN_THREAD(c);
    if (c >= pc.ChainCount) return;
    device const uint *record = ctx.Runs.Run(pc.Records) + 3u * c;
    device const uint *pairs = ctx.Runs.Run(pc.Output) + 2u * record[0];
    const uint n = record[1];
    ChainSummary summary{};
    const float3 origin = ctx.Position(pairs[0]);
    float3 normal = float3(0.f), centroid = float3(0.f);
    for (uint k = 0u; k < n; ++k) {
        const uint v = pairs[2u * k], source = pairs[2u * k + 1u], before = pairs[2u * ((k + n - 1u) % n)];
        if (source != InvalidOffset && ctx.View.HalfedgeFace(source) != InvalidOffset) summary.Winding += ctx.View.From(source) == before ? 1 : -1;
        const float3 p = ctx.Position(v);
        centroid += p;
        if (k + 1u < n && k > 0u) normal += cross(p - origin, ctx.Position(pairs[2u * (k + 1u)]) - origin);
    }
    for (const auto item : ctx.View.Fan(pairs[0])) {
        if (item.y == InvalidOffset || summary.MatchesFace) continue;
        const uint2 range = ctx.View.FaceHalfedges(item.y);
        if (range.y - range.x != n) continue;
        bool all = true;
        for (uint h = range.x; h < range.y && all; ++h) {
            const uint v = ctx.View.VertexOf(h);
            bool member = false;
            for (uint k = 0u; k < n && !member; ++k) member = pairs[2u * k] == v;
            all = member;
        }
        if (all) summary.MatchesFace = 1u;
    }
    summary.Normal = vec3(normal);
    summary.Centroid = vec3(centroid / float(n));
    summary.First = vec3(origin);
    summary.Last = vec3(ctx.Position(pairs[2u * (n - 1u)]));
    reinterpret_cast<device ChainSummary *>(ctx.Runs.Run(pc.Summaries))[c] = summary;
}

// The vertex of the chain nearest the point, the lowest at a tie.
kernel void SelectedChainNearest(JOB_TABLE_ARGS(SelectedChainsJob)) {
    CHAIN_THREAD(i);
    if (i) return;
    device const uint *record = ctx.Runs.Run(pc.Records);
    device const uint *pairs = ctx.Runs.Run(pc.Output) + 2u * record[0];
    uint nearest = 0u;
    float best = INFINITY;
    for (uint k = 0u; k < record[1]; ++k) {
        const float3 delta = ctx.Position(pairs[2u * k]) - float3(pc.Point);
        const float distance = dot(delta, delta);
        if (distance < best) {
            best = distance;
            nearest = k;
        }
    }
    ctx.Runs.Run(pc.Nearest)[0] = nearest;
}

// A point cloud's angles about its center in the plane of its farthest point and the farthest point across that, as Blender orders a vertex fill.
// The flag after the angles clears when the cloud is a point or a line.
kernel void SelectedChainPointAngles(JOB_TABLE_ARGS(SelectedChainsJob)) {
    JOB_TILE(pc);
    const ChainContext ctx{b, table, pc};
    device const uint *points = ctx.Runs.Run(pc.Points);
    device uint *angles = ctx.Runs.Run(pc.Angles);
    threadgroup float3 frame[3];
    threadgroup uint valid;
    if (lane == 0u) {
        float3 center = float3(0.f);
        for (uint k = 0u; k < pc.PointCount; ++k) center += ctx.Position(points[k]) / float(pc.PointCount);
        float3 tangent = float3(0.f);
        for (uint k = 0u; k < pc.PointCount; ++k) {
            const float3 delta = ctx.Position(points[k]) - center;
            if (dot(delta, delta) > dot(tangent, tangent)) tangent = delta;
        }
        valid = dot(tangent, tangent) == 0.f ? 0u : 1u;
        if (valid) tangent = normalize(tangent);
        float3 across = float3(0.f);
        for (uint k = 0u; k < pc.PointCount && valid; ++k) {
            float3 delta = ctx.Position(points[k]) - center;
            delta -= tangent * dot(delta, tangent);
            if (dot(delta, delta) > dot(across, across)) across = delta;
        }
        if (valid && dot(across, across) < 1e-20f) valid = 0u;
        if (valid) across = normalize(across);
        frame[0] = center;
        frame[1] = tangent;
        frame[2] = across;
        angles[pc.PointCount] = valid;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!valid) return;
    for (uint k = lane; k < pc.PointCount; k += ScanTileSize) {
        const float3 delta = ctx.Position(points[k]) - frame[0];
        angles[k] = as_type<uint>(atan2(dot(delta, frame[2]), dot(delta, frame[1])));
    }
}
