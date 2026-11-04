#include "ConnectivityRead.metal"
#include "ListRank.metal"
#include "gpu/BoundaryLoopsPushConstants.h"

// Each candidate links to the next boundary halfedge of its sheet, and pointer jumping labels and ranks the loops.
struct LoopContext {
    device const BindlessSet &B;
    constant BoundaryLoopsPushConstants &Pc;
    ConnectivityView View;

    LoopContext(device const BindlessSet &b, constant BoundaryLoopsPushConstants &pc)
        : B(b), Pc(pc), View{b, pc.Connectivity} {}
    device uint *Run(uint offset) const { return BindlessBufferMutable(uint, B.Buffer, Pc.ScratchSlot) + offset; }
    uint Handle(uint i) const { return BlockHandle(i, B, Pc.Blocks); }
    bool Listed(uint edge) const { return SortedContains(Run(Pc.Edges), Pc.EdgeCount, edge); }
    bool IsCandidate(uint h) const { return h != InvalidOffset && Run(Pc.Candidate)[h] != 0u; }
    uint SoleStart(uint h) const {
        const uint to = View.VertexOf(h);
        return Run(Pc.Outgoing)[to] == 1u ? Run(Pc.Outgoing)[Pc.VertexBound + to] : InvalidOffset;
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

// Each candidate marks itself and registers as a start at its origin vertex.
kernel void BoundaryLoopStarts(
    uint i [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant BoundaryLoopsPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const LoopContext ctx{b, pc};
    const uint h = ctx.Handle(i);
    if (h == InvalidOffset) return;
    const uint face = ctx.View.HalfedgeFace(h);
    if (face == InvalidOffset || ctx.View.Opposite(h) != InvalidOffset) return;
    if (pc.EdgeCount && !ctx.Listed(ctx.View.Edge(h))) return;
    const uint2 range = ctx.View.FaceHalfedges(face);
    bool joined = false;
    for (uint k = range.x; k < range.y; ++k) joined = joined || ctx.View.Opposite(k) != InvalidOffset;
    if (!joined) return;
    ctx.Run(pc.Candidate)[h] = 1u;
    const uint from = ctx.View.From(h);
    atomic_fetch_add_explicit(AtomicWord(ctx.Run(pc.Outgoing) + from), 1u, memory_order_relaxed);
    atomic_fetch_min_explicit(AtomicWord(ctx.Run(pc.Outgoing) + pc.VertexBound + from), h, memory_order_relaxed);
}

// Each candidate finds its sheet's successor and claims it when it is a candidate.
kernel void BoundaryLoopLink(
    uint i [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant BoundaryLoopsPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const LoopContext ctx{b, pc};
    const uint h = ctx.Handle(i);
    if (!ctx.IsCandidate(h)) return;
    const uint next = ctx.Successor(h);
    ctx.Run(pc.Natural)[h] = next;
    if (ctx.IsCandidate(next)) atomic_fetch_min_explicit(AtomicWord(ctx.Run(pc.Claim) + next), h, memory_order_relaxed);
}

// A candidate whose successor is not a candidate falls back to the sole candidate start at its end.
// The fallback applies when no successor claims that start, and the lowest such candidate wins it.
kernel void BoundaryLoopFallback(
    uint i [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant BoundaryLoopsPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const LoopContext ctx{b, pc};
    const uint h = ctx.Handle(i);
    if (!ctx.IsCandidate(h) || ctx.IsCandidate(ctx.Run(pc.Natural)[h])) return;
    const uint start = ctx.SoleStart(h);
    if (start == InvalidOffset || ctx.Run(pc.Claim)[start] != InvalidOffset) return;
    atomic_fetch_min_explicit(AtomicWord(ctx.Run(pc.Fallback) + start), h, memory_order_relaxed);
}

// Each live halfedge links to its successor or its won fallback, or to nothing, and starts its label at itself and its jump at its link.
kernel void BoundaryLoopNext(
    uint i [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant BoundaryLoopsPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const LoopContext ctx{b, pc};
    const uint h = ctx.Handle(i);
    if (h == InvalidOffset) return;
    uint next = InvalidOffset;
    if (ctx.IsCandidate(h)) {
        const uint natural = ctx.Run(pc.Natural)[h];
        if (ctx.IsCandidate(natural)) {
            next = natural;
        } else {
            const uint start = ctx.SoleStart(h);
            if (start != InvalidOffset && ctx.Run(pc.Fallback)[start] == h) next = start;
        }
    }
    ctx.Run(pc.Next)[h] = next;
    ctx.Run(pc.Labels)[h] = h;
    ctx.Run(pc.Jumps)[h] = next;
}

kernel void BoundaryLoopLabel(
    uint i [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant BoundaryLoopsPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const LoopContext ctx{b, pc};
    const uint h = ctx.Handle(i);
    if (h != InvalidOffset) LabelRound(ctx.Run(pc.Labels), ctx.Run(pc.Jumps), pc.HalfedgeBound, pc.Parameter & 1u, h);
}

// A candidate whose jumps never ran out lies on a loop.
// The loop breaks before its representative, its lowest halfedge, so list ranking counts each member's steps to the break.
kernel void BoundaryLoopRankInit(
    uint i [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant BoundaryLoopsPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const LoopContext ctx{b, pc};
    const uint h = ctx.Handle(i);
    if (h == InvalidOffset) return;
    const uint in = pc.Parameter * pc.HalfedgeBound;
    const uint label = ctx.Run(pc.Labels)[in + h], next = ctx.Run(pc.Next)[h];
    const bool closed = ctx.IsCandidate(h) && ctx.Run(pc.Jumps)[in + h] != InvalidOffset;
    ctx.Run(pc.Closed)[h] = closed ? 1u : 0u;
    ctx.Run(pc.Distances)[h] = 1u;
    ctx.Run(pc.Successors)[h] = !closed || next == label ? InvalidOffset : next;
}
kernel void BoundaryLoopRank(
    uint i [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant BoundaryLoopsPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const LoopContext ctx{b, pc};
    const uint h = ctx.Handle(i);
    if (h != InvalidOffset) RankRound(ctx.Run(pc.Distances), ctx.Run(pc.Successors), pc.HalfedgeBound, pc.Parameter & 1u, h);
}

// Each representative of a loop of at least three halfedges within the side limit counts its loop and its length for the scans, with a zero terminator.
// Parameter holds the final buffers' parity.
kernel void BoundaryLoopCount(
    uint i [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant BoundaryLoopsPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    if (i > pc.HalfedgeBound) return;
    const LoopContext ctx{b, pc};
    device uint *lengths = ctx.Run(pc.Lengths), *flags = ctx.Run(pc.Flags);
    if (i == pc.HalfedgeBound) {
        lengths[i] = flags[i] = 0u;
        return;
    }
    const uint converged = pc.Parameter * pc.HalfedgeBound;
    const bool representative = ctx.Run(pc.Candidate)[i] != 0u && ctx.Run(pc.Closed)[i] != 0u && ctx.Run(pc.Labels)[converged + i] == i;
    const uint length = representative ? ctx.Run(pc.Distances)[converged + i] : 0u;
    const bool kept = representative && length >= 3u && (!pc.MaxSides || length <= pc.MaxSides);
    lengths[i] = kept ? length : 0u;
    flags[i] = kept ? 1u : 0u;
}

// Each member of a kept loop writes its origin vertex and itself at its place in face winding order, which runs against the halfedges.
kernel void BoundaryLoopOutput(
    uint i [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant BoundaryLoopsPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const LoopContext ctx{b, pc};
    const uint h = ctx.Handle(i);
    if (h == InvalidOffset || ctx.Run(pc.Closed)[h] == 0u) return;
    const uint converged = pc.Parameter * pc.HalfedgeBound;
    const uint label = ctx.Run(pc.Labels)[converged + h];
    device const uint *lengths = ctx.Run(pc.Lengths), *flags = ctx.Run(pc.Flags);
    if (flags[label + 1u] == flags[label]) return;
    const uint offset = lengths[label];
    if (label == h) {
        device uint *records = ctx.Run(pc.Records) + 2u * flags[h];
        records[0] = offset;
        records[1] = ctx.Run(pc.Distances)[converged + h];
    }
    device uint *pair = ctx.Run(pc.Output) + 2u * (offset + ctx.Run(pc.Distances)[converged + h] - 1u);
    pair[0] = ctx.View.From(h);
    pair[1] = h;
}
