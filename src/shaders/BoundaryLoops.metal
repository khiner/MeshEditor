#include "ConnectivityRead.metal"
#include "ListRank.metal"
#include "gpu/BoundaryLoopsJob.h"

// Each candidate links to the next boundary halfedge of its sheet, and pointer jumping labels and ranks the loops.
struct LoopContext {
    device const BindlessSet &B;
    constant BoundaryLoopsJob &Pc;
    ConnectivityView View;
    JobRuns Runs;

    LoopContext(device const BindlessSet &b, constant JobTablePushConstants &table, constant BoundaryLoopsJob &pc) : B(b), Pc(pc), View{b, pc.Connectivity}, Runs(ScratchOf(b, table)) {}
    // Whether each halfedge's links close into a loop, in the jump half the label rounds of final parity `parity` leave stale.
    device uint *Closed(uint parity) const { return Runs.Run(Pc.Jumps + (1u - parity) * Pc.HalfedgeBound); }
    uint Handle(uint i) const { return BlockHandle(i, B, Pc.Blocks); }
    bool Listed(uint edge) const { return SortedContains(Runs.Run(Pc.Edges), Pc.EdgeCount, edge); }
    bool IsCandidate(uint h) const { return h != InvalidOffset && Runs.Run(Pc.Candidate)[h] != 0u; }
    uint SoleStart(uint h) const {
        const uint to = View.VertexOf(h);
        return Runs.Run(Pc.Outgoing)[to] == 1u ? Runs.Run(Pc.Outgoing)[Pc.VertexBound + to] : InvalidOffset;
    }
    uint Across(uint h) const {
        if (h == InvalidOffset) return InvalidOffset;
        const uint opposite = View.Opposite(h);
        return opposite == InvalidOffset ? InvalidOffset : View.Next(opposite);
    }
    // The next boundary halfedge of h's sheet, around the faces at h's end to the first halfedge without an opposite, or none when that fan closes.
    uint Successor(uint h) const {
        uint next = View.Next(h), fast = next;
        while (next != InvalidOffset && View.Opposite(next) != InvalidOffset) {
            next = Across(next);
            fast = Across(Across(fast));
            if (fast != InvalidOffset && next == fast) return InvalidOffset;
        }
        return next;
    }
};
// Declares a job table kernel's context, its job `pc` and the thread's index `i` within the job.
#define LOOP_CONTEXT JOB_THREAD(pc, i); const LoopContext ctx{b, table, pc}

// Each candidate marks itself and registers as a start at its origin vertex.
kernel void BoundaryLoopStarts(JOB_TABLE_ARGS(BoundaryLoopsJob)) {
    LOOP_CONTEXT;
    const uint h = ctx.Handle(i);
    if (h == InvalidOffset) return;
    const uint face = ctx.View.HalfedgeFace(h);
    if (face == InvalidOffset || ctx.View.Opposite(h) != InvalidOffset) return;
    if (pc.EdgeCount && !ctx.Listed(ctx.View.Edge(h))) return;
    const uint2 range = ctx.View.FaceHalfedges(face);
    bool joined = false;
    for (uint k = range.x; k < range.y; ++k) joined = joined || ctx.View.Opposite(k) != InvalidOffset;
    if (!joined) return;
    ctx.Runs.Run(pc.Candidate)[h] = 1u;
    const uint from = ctx.View.From(h);
    atomic_fetch_add_explicit(AtomicWord(ctx.Runs.Run(pc.Outgoing) + from), 1u, memory_order_relaxed);
    atomic_fetch_min_explicit(AtomicWord(ctx.Runs.Run(pc.Outgoing) + pc.VertexBound + from), h, memory_order_relaxed);
}

// Each candidate finds its sheet's successor and claims it when it is a candidate.
kernel void BoundaryLoopLink(JOB_TABLE_ARGS(BoundaryLoopsJob)) {
    LOOP_CONTEXT;
    const uint h = ctx.Handle(i);
    if (!ctx.IsCandidate(h)) return;
    const uint next = ctx.Successor(h);
    ctx.Runs.Run(pc.Natural)[h] = next;
    if (ctx.IsCandidate(next)) atomic_fetch_min_explicit(AtomicWord(ctx.Runs.Run(pc.Claim) + next), h, memory_order_relaxed);
}

// A candidate whose successor is not a candidate falls back to the sole candidate start at its end.
// The fallback applies when no successor claims that start, and the lowest such candidate wins it.
kernel void BoundaryLoopFallback(JOB_TABLE_ARGS(BoundaryLoopsJob)) {
    LOOP_CONTEXT;
    const uint h = ctx.Handle(i);
    if (!ctx.IsCandidate(h) || ctx.IsCandidate(ctx.Runs.Run(pc.Natural)[h])) return;
    const uint start = ctx.SoleStart(h);
    if (start == InvalidOffset || ctx.Runs.Run(pc.Claim)[start] != InvalidOffset) return;
    atomic_fetch_min_explicit(AtomicWord(ctx.Runs.Run(pc.Fallback) + start), h, memory_order_relaxed);
}

// Each live halfedge links to its successor or its won fallback, or to nothing, and starts its label at itself and its jump at its link.
kernel void BoundaryLoopNext(JOB_TABLE_ARGS(BoundaryLoopsJob)) {
    LOOP_CONTEXT;
    const uint h = ctx.Handle(i);
    if (h == InvalidOffset) return;
    uint next = InvalidOffset;
    if (ctx.IsCandidate(h)) {
        const uint natural = ctx.Runs.Run(pc.Natural)[h];
        if (ctx.IsCandidate(natural)) {
            next = natural;
        } else {
            const uint start = ctx.SoleStart(h);
            if (start != InvalidOffset && ctx.Runs.Run(pc.Fallback)[start] == h) next = start;
        }
    }
    ctx.Runs.Run(pc.Next)[h] = next;
    ctx.Runs.Run(pc.Labels)[h] = h;
    ctx.Runs.Run(pc.Jumps)[h] = next;
}

kernel void BoundaryLoopLabel(JOB_TABLE_ARGS(BoundaryLoopsJob)) {
    LOOP_CONTEXT;
    const uint h = ctx.Handle(i);
    if (h != InvalidOffset) LabelRound(ctx.Runs.Run(pc.Labels), ctx.Runs.Run(pc.Jumps), pc.HalfedgeBound, table.Parameter & 1u, h);
}

// A candidate whose jumps never ran out lies on a loop.
// The loop breaks before its representative, its lowest halfedge, so list ranking counts each member's steps to the break.
kernel void BoundaryLoopRankInit(JOB_TABLE_ARGS(BoundaryLoopsJob)) {
    LOOP_CONTEXT;
    const uint h = ctx.Handle(i);
    if (h == InvalidOffset) return;
    const uint in = table.Parameter * pc.HalfedgeBound;
    const uint label = ctx.Runs.Run(pc.Labels)[in + h], next = ctx.Runs.Run(pc.Next)[h];
    const bool closed = ctx.IsCandidate(h) && ctx.Runs.Run(pc.Jumps)[in + h] != InvalidOffset;
    ctx.Closed(table.Parameter)[h] = closed ? 1u : 0u;
    ctx.Runs.Run(pc.Distances)[h] = 1u;
    ctx.Runs.Run(pc.Successors)[h] = !closed || next == label ? InvalidOffset : next;
}
kernel void BoundaryLoopRank(JOB_TABLE_ARGS(BoundaryLoopsJob)) {
    LOOP_CONTEXT;
    const uint h = ctx.Handle(i);
    if (h != InvalidOffset) RankRound(ctx.Runs.Run(pc.Distances), ctx.Runs.Run(pc.Successors), pc.HalfedgeBound, table.Parameter & 1u, h);
}

// Each representative of a loop of at least three halfedges within the side limit counts its loop and its length for the scans, with a zero terminator, which writes the parity.
// Parameter holds the final buffers' parity, one thread per halfedge handle and one for the terminator.
kernel void BoundaryLoopCount(JOB_TABLE_ARGS(BoundaryLoopsJob)) {
    LOOP_CONTEXT;
    if (i > pc.HalfedgeBound) return;
    device uint *lengths = ctx.Runs.Run(pc.Lengths), *flags = ctx.Runs.Run(pc.Flags);
    if (i == pc.HalfedgeBound) {
        lengths[i] = flags[i] = 0u;
        ctx.Runs.Run(pc.Parity)[0] = table.Parameter;
        return;
    }
    const uint converged = table.Parameter * pc.HalfedgeBound;
    const bool representative = ctx.Runs.Run(pc.Candidate)[i] != 0u && ctx.Closed(table.Parameter)[i] != 0u && ctx.Runs.Run(pc.Labels)[converged + i] == i;
    const uint length = representative ? ctx.Runs.Run(pc.Distances)[converged + i] : 0u;
    const bool kept = representative && length >= 3u && (!pc.MaxSides || length <= pc.MaxSides);
    lengths[i] = kept ? length : 0u;
    flags[i] = kept ? 1u : 0u;
}

// Each member of a kept loop writes its origin vertex and itself at its place in face winding order, which runs against the halfedges.
kernel void BoundaryLoopOutput(JOB_TABLE_ARGS(BoundaryLoopsJob)) {
    LOOP_CONTEXT;
    const uint h = ctx.Handle(i);
    const uint parity = ctx.Runs.Run(pc.Parity)[0];
    if (h == InvalidOffset || ctx.Closed(parity)[h] == 0u) return;
    const uint converged = parity * pc.HalfedgeBound;
    const uint label = ctx.Runs.Run(pc.Labels)[converged + h];
    device const uint *lengths = ctx.Runs.Run(pc.Lengths), *flags = ctx.Runs.Run(pc.Flags);
    if (flags[label + 1u] == flags[label]) return;
    const uint offset = lengths[label];
    if (label == h) {
        device uint *records = ctx.Runs.Run(pc.Records) + 2u * flags[h];
        records[0] = offset;
        records[1] = ctx.Runs.Run(pc.Distances)[converged + h];
    }
    device uint *pair = ctx.Runs.Run(pc.Output) + 2u * (offset + ctx.Runs.Run(pc.Distances)[converged + h] - 1u);
    pair[0] = ctx.View.From(h);
    pair[1] = h;
}
