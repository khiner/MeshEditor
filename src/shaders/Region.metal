// IEEE arithmetic without contraction keeps the arrangement predicates consistent.
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)

// The fill of a plane arrangement.
#include "BooleanCollider.metal"
#include "BooleanShadows.metal"
#include "ConnectivityRead.metal"
#include "ElementWorkShared.metal"
#include "ListRank.metal"
#include "OrderedFloat.metal"
#include "gpu/Region.h"

// Segment normals agreeing within this count as one direction for a vertex's move.
constant float RegionParallelTolerance = 1e-6f;

using RegionContext = JobContext<RegionJob>;

// The plane geometry at z zero, with the first operand's vertices moving an eighth further than the second's and shifted per axis.
struct RegionGeometry {
    device const packed_float2 *Positions, *Moves;
    device const RegionSegment *Segments;
    uint FirstSecond;
    float Tol;

    float3 Position(uint v) const { return float3(float2(Positions[v]), 0.f); }
    float3 Move(uint v) const {
        const bool first = v < FirstSecond;
        return float3(float2(Moves[v]) * (first ? 1.125f : 1.f) + (first ? float2(1.f / 61.f, 1.f / 89.f) : float2(0.f)), 0.f);
    }
    float2 Direction(uint s) const { return float2(Positions[Segments[s].End]) - float2(Positions[Segments[s].Start]); }
};
inline RegionGeometry Geometry(RegionContext ctx, RegionJob job) {
    device const uint *bounds = ctx.Run<uint>(job.Bounds);
    float scale = INFINITY;
    for (uint side = 0u; side < 2u; ++side) {
        if (!job.Segments[side]) continue;
        float largest = 0.f;
        for (uint k = 0u; k < 4u; ++k) largest = max(largest, abs(OrderedFloat(bounds[4u * side + k])));
        scale = min(scale, largest);
    }
    return {ctx.Run<packed_float2>(job.Positions), ctx.Run<packed_float2>(job.Moves), ctx.Run<RegionSegment>(job.SegmentRecords), job.Vertices[0], BoolTolerance(scale)};
}
// The winding a segment of direction d adds to a point below it that it shadows in x, when the ray from the point along y crosses it.
inline int RayContribution(float2 d) { return d.x > 0.f ? -1 : d.x < 0.f ? 1 : 0; }
inline float Cross2(float2 a, float2 b) { return a.x * b.y - a.y * b.x; }
// Whether a point displaced from a vertex of segment L by u, then by far less along `tie`, lies below L.
inline bool Below(float2 d_l, float2 u, float2 tie) {
    float side = Cross2(d_l, u);
    if (side == 0.f) side = Cross2(d_l, tie);
    return d_l.x > 0.f ? side < 0.f : side > 0.f;
}
inline bool Inside(RegionFill fill, int2 w) {
    switch (fill) {
        case RegionFill::Positive: return w.x > 0;
        case RegionFill::Union: return w.x > 0 || w.y > 0;
        case RegionFill::Difference: return w.x > 0 && w.y <= 0;
        case RegionFill::Intersection: return w.x > 0 && w.y > 0;
    }
}
inline uint PieceFirst(RegionContext ctx, RegionJob job, uint s) { return ctx.Run<uint>(job.CrossingFirst)[s] + s; }
inline uint SegmentOf(RegionContext ctx, RegionJob job, uint p) {
    return PartitionPoint(job.Segments[0] + job.Segments[1], [&](uint k) { return PieceFirst(ctx, job, k) <= p; }) - 1u;
}
inline bool PieceIsKept(RegionContext ctx, RegionJob job, uint p) {
    device const uint *kept = ctx.Run<uint>(job.PieceKept);
    return kept[p] != kept[p + 1u];
}
inline float2 PiecePoint(RegionContext ctx, RegionJob job, RegionGeometry g, uint s, uint k, bool end) {
    const uint first = ctx.Run<uint>(job.CrossingFirst)[s], count = ctx.Run<uint>(job.CrossingFirst)[s + 1u] - first;
    if (end ? k == count : k == 0u) return g.Position(end ? g.Segments[s].End : g.Segments[s].Start).xy;
    return float2(ctx.Run<RegionCrossing>(job.Crossings)[first + (end ? k : k - 1u)].Position);
}

// The first-order move of the parameter along s of its crossing with t at parameter lambda, as the moves carry both segments.
inline float ParameterShift(RegionGeometry g, uint s, uint t, float lambda) {
    const RegionSegment a = g.Segments[s], c = g.Segments[t];
    const float2 origin = g.Position(a.Start).xy, u = g.Direction(s), start = g.Position(c.Start).xy, v = g.Direction(t);
    const float2 p = g.Move(a.Start).xy, du = g.Move(a.End).xy - p, r = g.Move(c.Start).xy, dv = g.Move(c.End).xy - r;
    const float uv = Cross2(u, v);
    if (uv == 0.f) return 0.f;
    const float mu = Cross2(start - origin, u) / uv;
    return (Cross2(r - p, v) + mu * Cross2(dv, v) - lambda * Cross2(du, v)) / uv;
}
// Parameters farther apart than `tolerance` decide, then the parameters' moves, then the other segment.
inline bool CrossingBefore(RegionCrossing a, RegionCrossing c, float tolerance) {
    if (abs(a.Parameter - c.Parameter) > tolerance) return a.Parameter < c.Parameter;
    if (a.Shift != c.Shift) return a.Shift < c.Shift;
    return a.Other < c.Other;
}

inline float3 RecordPosition(device const BindlessSet &b, constant RegionSegmentsPushConstants &pc, ConnectivityView view, uint corner) {
    const float3 p = float3(BindlessBuffer(Vertex, b.VertexBuffer, pc.VertexSlot)[view.VertexAddress(view.VertexOf(corner))].Position);
    return pc.Rotation.Unpack() * p + float3(pc.Translation);
}
inline float2 PlaneCoordinates(constant RegionSegmentsPushConstants &pc, float3 d) { return float2(dot(d, float3(pc.U)), dot(d, float3(pc.V))); }
inline float2 PlanePosition(device const BindlessSet &b, constant RegionSegmentsPushConstants &pc, ConnectivityView view, uint corner) {
    return PlaneCoordinates(pc, RecordPosition(b, pc, view, corner) - float3(pc.Point));
}
// Whether a live face's halfedge is selected, which is always unless the job takes cusps only.
inline bool Selected(device const BindlessSet &b, constant RegionSegmentsPushConstants &pc, ConnectivityView view, uint h) {
    if (!pc.Cusps) return true;
    const uint face = view.HalfedgeFace(h);
    const float3 normal = float3(pc.Normal);
    device const packed_float3 *normals = BindlessBuffer(packed_float3, b.Buffer, pc.FaceNormalSlot);
    if (dot(pc.Rotation.Unpack() * float3(normals[view.FaceAddress(face)]), normal) < 0.f) return false;
    const uint opposite = view.Opposite(h);
    if (opposite == InvalidOffset) return true;
    const uint other = view.HalfedgeFace(opposite);
    return other == InvalidOffset || dot(pc.Rotation.Unpack() * float3(normals[view.FaceAddress(other)]), normal) < 0.f;
}
inline device uint *Aliases(device const BindlessSet &b, constant RegionSegmentsPushConstants &pc) { return BindlessBufferMutable(uint, b.Buffer, pc.Aliases.Slot) + pc.Aliases.Offset; }
inline uint Resolve(device const uint *aliases, uint v) {
    for (uint n = 0u; n < 64u && aliases[v] != v; ++n) v = aliases[v];
    return v;
}
// A selected halfedge's segment ends, as job vertices after merging, or equal ends when it projects to a point.
inline uint2 SegmentEnds(device const BindlessSet &b, constant RegionSegmentsPushConstants &pc, ConnectivityView view, uint h) {
    device const uint *aliases = Aliases(b, pc);
    return uint2(Resolve(aliases, pc.VertexBase + view.From(h)), Resolve(aliases, pc.VertexBase + view.VertexOf(h)));
}
template<typename Visit> inline void ForEachFaceSegment(device const BindlessSet &b, constant RegionSegmentsPushConstants &pc, ConnectivityView view, uint f, Visit visit) {
    const uint2 range = view.FaceHalfedges(ElementWorkDomain{b, pc.Faces}.Handle(f));
    for (uint h = range.x; h < range.y; ++h) {
        if (!Selected(b, pc, view, h)) continue;
        const uint2 ends = SegmentEnds(b, pc, view, h);
        if (ends.x != ends.y) visit(h, ends);
    }
}

// Each selected halfedge projecting to a point merges its ends into the lower vertex, so the segments on either side meet at one arrangement vertex.
kernel void RegionSegmentAlias(
    uint f [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant RegionSegmentsPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    if (f >= pc.FaceCount) return;
    const ConnectivityView view{b, pc.Connectivity};
    const uint2 range = view.FaceHalfedges(ElementWorkDomain{b, pc.Faces}.Handle(f));
    device atomic_uint *aliases = AtomicWord(Aliases(b, pc));
    for (uint h = range.x; h < range.y; ++h) {
        const uint previous = view.Previous(h);
        if (!Selected(b, pc, view, h) || any(PlanePosition(b, pc, view, previous) != PlanePosition(b, pc, view, h))) continue;
        const uint from = pc.VertexBase + view.VertexOf(previous), to = pc.VertexBase + view.VertexOf(h);
        atomic_fetch_min_explicit(&aliases[max(from, to)], min(from, to), memory_order_relaxed);
    }
}
// Each live face counts its selected halfedges with two distinct ends and writes their ends' plane positions and the operand's bounds.
kernel void RegionSegmentCount(
    uint f [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant RegionSegmentsPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    if (f > pc.FaceCount) return;
    device uint *counts = BindlessBufferMutable(uint, b.Buffer, pc.Counts.Slot) + pc.Counts.Offset;
    if (f == pc.FaceCount) {
        counts[f] = 0u;
        return;
    }
    const ConnectivityView view{b, pc.Connectivity};
    device packed_float2 *positions = reinterpret_cast<device packed_float2 *>(BindlessBufferMutable(uint, b.Buffer, pc.Positions.Slot) + pc.Positions.Offset);
    device atomic_uint *bounds = AtomicWord(BindlessBufferMutable(uint, b.Buffer, pc.Bounds.Slot) + pc.Bounds.Offset) + 4u * pc.Operand;
    device atomic_uint *deviation = AtomicWord(BindlessBufferMutable(uint, b.Buffer, pc.Deviation.Slot) + pc.Deviation.Offset);
    uint count = 0u;
    ForEachFaceSegment(b, pc, view, f, [&](uint h, uint2 ends) {
        ++count;
        const uint corners[2] = {view.Previous(h), h};
        for (uint i = 0u; i < 2u; ++i) {
            const float3 d = RecordPosition(b, pc, view, corners[i]) - float3(pc.Point);
            const float2 p = PlaneCoordinates(pc, d);
            positions[ends[i]] = packed_float2(p);
            atomic_fetch_min_explicit(&bounds[0], OrderedBits(p.x), memory_order_relaxed);
            atomic_fetch_min_explicit(&bounds[1], OrderedBits(p.y), memory_order_relaxed);
            atomic_fetch_max_explicit(&bounds[2], OrderedBits(p.x), memory_order_relaxed);
            atomic_fetch_max_explicit(&bounds[3], OrderedBits(p.y), memory_order_relaxed);
            if (!pc.Cusps) atomic_fetch_max_explicit(deviation, OrderedBits(abs(dot(d, float3(pc.Normal)))), memory_order_relaxed);
        }
    });
    counts[f] = count;
}
kernel void RegionSegmentEmit(
    uint f [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant RegionSegmentsPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    if (f >= pc.FaceCount) return;
    device const uint *counts = BindlessBuffer(uint, b.Buffer, pc.Counts.Slot) + pc.Counts.Offset;
    if (counts[f] == counts[f + 1u]) return;
    const ConnectivityView view{b, pc.Connectivity};
    device RegionSegment *segments = reinterpret_cast<device RegionSegment *>(BindlessBufferMutable(uint, b.Buffer, pc.Segments.Slot) + pc.Segments.Offset);
    uint slot = pc.SegmentBase + counts[f];
    ForEachFaceSegment(b, pc, view, f, [&](uint, uint2 ends) {
        RegionSegment segment;
        segment.Start = ends.x;
        segment.End = ends.y;
        segment.Operand = pc.Operand;
        segments[slot++] = segment;
    });
}

kernel void RegionIncidentCount(
    uint s [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant JobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const RegionContext ctx{b, pc};
    const RegionJob job = ctx.Job();
    if (s >= job.Segments[0] + job.Segments[1]) return;
    const RegionSegment segment = ctx.Run<RegionSegment>(job.SegmentRecords)[s];
    device atomic_uint *first = ctx.Run<atomic_uint>(job.IncidentFirst);
    atomic_fetch_add_explicit(&first[segment.Start], 1u, memory_order_relaxed);
    atomic_fetch_add_explicit(&first[segment.End], 1u, memory_order_relaxed);
}
kernel void RegionIncidentCursors(
    uint v [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant JobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const RegionContext ctx{b, pc};
    const RegionJob job = ctx.Job();
    if (v >= job.Vertices[0] + job.Vertices[1]) return;
    ctx.Run<uint>(job.IncidentCursor)[v] = ctx.Run<uint>(job.IncidentFirst)[v];
}
kernel void RegionIncidentFill(
    uint s [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant JobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const RegionContext ctx{b, pc};
    const RegionJob job = ctx.Job();
    if (s >= job.Segments[0] + job.Segments[1]) return;
    const RegionSegment segment = ctx.Run<RegionSegment>(job.SegmentRecords)[s];
    device atomic_uint *cursors = ctx.Run<atomic_uint>(job.IncidentCursor);
    device uint *incident = ctx.Run<uint>(job.Incident);
    incident[atomic_fetch_add_explicit(&cursors[segment.Start], 1u, memory_order_relaxed)] = s;
    incident[atomic_fetch_add_explicit(&cursors[segment.End], 1u, memory_order_relaxed)] = s | RegionIncidentEnd;
}

// Each vertex's move shifts every incident segment outward by one unit, along one segment's outward normal or the offset of two.
kernel void RegionMoves(
    uint v [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant JobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const RegionContext ctx{b, pc};
    const RegionJob job = ctx.Job();
    if (v >= job.Vertices[0] + job.Vertices[1]) return;
    device const uint *first = ctx.Run<uint>(job.IncidentFirst), *incident = ctx.Run<uint>(job.Incident);
    device const packed_float2 *positions = ctx.Run<packed_float2>(job.Positions);
    device const RegionSegment *segments = ctx.Run<RegionSegment>(job.SegmentRecords);
    float2 normals[2];
    uint count = 0u;
    for (uint i = first[v]; i < first[v + 1u] && count < 2u; ++i) {
        const RegionSegment segment = segments[incident[i] & ~RegionIncidentEnd];
        const float2 d = float2(positions[segment.End]) - float2(positions[segment.Start]);
        if (!(dot(d, d) > 0.f)) continue;
        const float2 n = normalize(float2(d.y, -d.x));
        if (count == 1u && abs(dot(normals[0], n)) > 1.f - RegionParallelTolerance) continue;
        normals[count++] = n;
    }
    float2 move = float2(0.f);
    if (count == 1u) move = normals[0];
    else if (count == 2u) move = (normals[0] + normals[1]) / (1.f + dot(normals[0], normals[1]));
    if (dot(move, move) > 4.f) move = 2.f * normalize(move);
    ctx.Run<packed_float2>(job.Moves)[v] = packed_float2(move);
}

inline uint SpreadBits2(uint v) {
    v = (v | (v << 8u)) & 0x00FF00FFu;
    v = (v | (v << 4u)) & 0x0F0F0F0Fu;
    v = (v | (v << 2u)) & 0x33333333u;
    v = (v | (v << 1u)) & 0x55555555u;
    return v;
}
kernel void RegionMorton(
    uint s [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant JobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const RegionContext ctx{b, pc};
    const RegionJob job = ctx.Job();
    if (s >= job.Segments[0] + job.Segments[1]) return;
    const RegionGeometry g = Geometry(ctx, job);
    const float2 a = g.Position(g.Segments[s].Start).xy, c = g.Position(g.Segments[s].End).xy, center = (min(a, c) + max(a, c)) * .5f;
    device const uint *bounds = ctx.Run<uint>(job.Bounds);
    float2 low = float2(INFINITY), high = float2(-INFINITY);
    for (uint side = 0u; side < 2u; ++side) {
        if (!job.Segments[side]) continue;
        low = min(low, float2(OrderedFloat(bounds[4u * side]), OrderedFloat(bounds[4u * side + 1u])));
        high = max(high, float2(OrderedFloat(bounds[4u * side + 2u]), OrderedFloat(bounds[4u * side + 3u])));
    }
    uint code = 0u;
    for (uint axis = 0u; axis < 2u; ++axis) {
        const float extent = high[axis] - low[axis];
        const float unit = extent > 0.f ? (center[axis] - low[axis]) / extent : 0.f;
        code |= SpreadBits2(uint(clamp(65535.f * unit, 0.f, 65535.f))) << (1u - axis);
    }
    ctx.Run<uint>(job.Collider.Keys)[s] = code;
    ctx.Run<uint>(job.Collider.Order)[s] = s;
}
kernel void RegionRefit(
    uint leaf [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant JobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const RegionContext ctx{b, pc};
    const RegionJob job = ctx.Job();
    const ColliderRuns runs = job.Collider;
    if (leaf >= runs.Leaves) return;
    const RegionGeometry g = Geometry(ctx, job);
    const uint s = ctx.Run<uint>(runs.Order)[leaf];
    const float3 a = g.Position(g.Segments[s].Start), c = g.Position(g.Segments[s].End);
    RefitClimb(ctx, runs, leaf, min(a, c), max(a, c));
}

// Visits each crossing of segment s with another segment, computed in the lower segment's role so both segments see one crossing.
template<typename Visit>
inline uint SegmentCrossings(RegionContext ctx, RegionJob job, RegionGeometry g, uint s, Visit visit) {
    const RegionSegment segment = g.Segments[s];
    const float3 a = g.Position(segment.Start), c = g.Position(segment.End);
    const float3 low = min(a, c) - g.Tol, high = max(a, c) + g.Tol;
    uint count = 0u;
    BoolTraverse(TreeOf(ctx, job.Collider, 0u), [&](device const float *box) { return BoxOverlaps(box, low, high); }, [&](uint t) {
        if (t == s) return;
        const RegionSegment other = g.Segments[t];
        if (other.Start == segment.Start || other.Start == segment.End || other.End == segment.Start || other.End == segment.End) return;
        const BoolCrossXY crossing = s < t ? ShadowCrossing(g, segment.Start, segment.End, other.Start, other.End) : ShadowCrossing(g, other.Start, other.End, segment.Start, segment.End);
        if (crossing.S == 0) return;
        visit(count, t, s < t ? crossing.S : -crossing.S, crossing.XYZZ.xy);
        ++count;
    });
    return count;
}
kernel void RegionCrossingCount(
    uint s [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant JobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const RegionContext ctx{b, pc};
    const RegionJob job = ctx.Job();
    const uint segments = job.Segments[0] + job.Segments[1];
    if (s > segments) return;
    ctx.Run<uint>(job.CrossingFirst)[s] = s == segments ? 0u : SegmentCrossings(ctx, job, Geometry(ctx, job), s, [](uint, uint, int, float2) {});
}
kernel void RegionCrossingEmit(
    uint s [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant JobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const RegionContext ctx{b, pc};
    const RegionJob job = ctx.Job();
    if (s >= job.Segments[0] + job.Segments[1]) return;
    const RegionGeometry g = Geometry(ctx, job);
    const uint first = ctx.Run<uint>(job.CrossingFirst)[s];
    device RegionCrossing *entries = ctx.Run<RegionCrossing>(job.Crossings) + first;
    const float2 a = g.Position(g.Segments[s].Start).xy, d = g.Direction(s);
    const float length2 = dot(d, d);
    const uint count = SegmentCrossings(ctx, job, g, s, [&](uint i, uint t, int sign, float2 position) {
        RegionCrossing entry;
        entry.Other = t;
        entry.Delta = -sign;
        entry.Position = packed_float2(position);
        entry.Parameter = length2 > 0.f ? dot(position - a, d) / length2 : 0.f;
        entry.Shift = ParameterShift(g, s, t, entry.Parameter);
        entries[i] = entry;
    });
    const float tolerance = length2 > 0.f ? g.Tol * rsqrt(length2) : 0.f;
    InsertionSort(entries, count, [&](RegionCrossing a, RegionCrossing c) { return CrossingBefore(a, c, tolerance); });
}

// Each vertex with segments takes its operand windings in the moved configuration from the signed crossings of the other segments above it along y.
kernel void RegionWindings(
    uint v [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant JobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const RegionContext ctx{b, pc};
    const RegionJob job = ctx.Job();
    if (v >= job.Vertices[0] + job.Vertices[1]) return;
    device const uint *first = ctx.Run<uint>(job.IncidentFirst);
    if (first[v] == first[v + 1u]) return;
    const RegionGeometry g = Geometry(ctx, job);
    const float3 p = g.Position(v);
    const float3 low = float3(p.x - g.Tol, p.y - g.Tol, 0.f), high = float3(p.x + g.Tol, INFINITY, 0.f);
    int2 winding = int2(0);
    BoolTraverse(TreeOf(ctx, job.Collider, 0u), [&](device const float *box) { return BoxOverlaps(box, low, high); }, [&](uint t) {
        const RegionSegment segment = g.Segments[t];
        if (segment.Start == v || segment.End == v) return;
        const BoolShadow shadow = Shadow01(g, true, v, segment.Start, segment.End);
        if (shadow.S != 0) winding[segment.Operand] -= shadow.S;
    });
    ctx.Run<packed_int2>(job.Windings)[v] = packed_int2(winding);
}

// Each segment walks its pieces from the winding just right of its start, corrected for the segments meeting there, through each crossing's change.
kernel void RegionPieces(
    uint s [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant JobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const RegionContext ctx{b, pc};
    const RegionJob job = ctx.Job();
    const uint segments = job.Segments[0] + job.Segments[1];
    if (s > segments) return;
    device uint *kept = ctx.Run<uint>(job.PieceKept), *reversed = ctx.Run<uint>(job.PieceReversed);
    if (s == segments) {
        kept[job.Pieces] = 0u;
        return;
    }
    const RegionGeometry g = Geometry(ctx, job);
    const RegionSegment segment = g.Segments[s];
    const float2 d = g.Direction(s), outward = float2(d.y, -d.x);
    // The windings at the moved start vertex count none of the segments meeting there.
    int2 right = int2(ctx.Run<packed_int2>(job.Windings)[segment.Start]);
    device const uint *first = ctx.Run<uint>(job.IncidentFirst), *incident = ctx.Run<uint>(job.Incident);
    const float3 start = g.Position(segment.Start);
    const float ux = d.x != 0.f ? d.x : outward.x;
    for (uint i = first[segment.Start]; i < first[segment.Start + 1u]; ++i) {
        const uint l = incident[i] & ~RegionIncidentEnd;
        const RegionSegment other = g.Segments[l];
        const uint w = other.Start == segment.Start ? other.End : other.Start;
        float wx = g.Position(w).x - start.x;
        if (wx == 0.f) wx = g.Move(w).x - g.Move(segment.Start).x;
        if (wx == 0.f || (ux > 0.f) != (wx > 0.f)) continue;
        const float2 dl = g.Direction(l);
        if (Below(dl, d, outward)) right[other.Operand] += RayContribution(dl);
    }
    const uint crossing_first = ctx.Run<uint>(job.CrossingFirst)[s], count = ctx.Run<uint>(job.CrossingFirst)[s + 1u] - crossing_first;
    device const RegionCrossing *entries = ctx.Run<RegionCrossing>(job.Crossings) + crossing_first;
    const uint piece_first = crossing_first + s;
    for (uint k = 0u; k <= count; ++k) {
        int2 left = right;
        left[segment.Operand] += 1;
        const bool in_left = Inside(job.Fill, left), in_right = Inside(job.Fill, right);
        kept[piece_first + k] = in_left != in_right ? 1u : 0u;
        reversed[piece_first + k] = in_right ? 1u : 0u;
        if (k < count) right[g.Segments[entries[k].Other].Operand] += entries[k].Delta;
    }
}

kernel void RegionKeptFill(
    uint p [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant JobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const RegionContext ctx{b, pc};
    const RegionJob job = ctx.Job();
    if (p >= job.Pieces || !PieceIsKept(ctx, job, p)) return;
    ctx.Run<uint>(job.Kept)[ctx.Run<uint>(job.PieceKept)[p]] = p;
}

struct RegionTravel {
    uint Segment, Index, Count;
    bool Reversed;
};
inline RegionTravel TravelOf(RegionContext ctx, RegionJob job, uint p) {
    const uint s = SegmentOf(ctx, job, p);
    device const uint *first = ctx.Run<uint>(job.CrossingFirst);
    return {s, p - PieceFirst(ctx, job, s), first[s + 1u] - first[s], ctx.Run<uint>(job.PieceReversed)[p] != 0u};
}
// Of the candidate pieces leaving the point a kept piece arrives at, the turn keeps the one first clockwise from the arrival's reverse.
struct RegionTurn {
    float2 Back;
    float Best;
    uint Piece;
};
inline void Consider(RegionContext ctx, RegionJob job, RegionGeometry g, thread RegionTurn &turn, uint s, uint k, bool must_reverse) {
    const uint p = PieceFirst(ctx, job, s) + k;
    if (!PieceIsKept(ctx, job, p) || (ctx.Run<uint>(job.PieceReversed)[p] != 0u) != must_reverse) return;
    const float2 d = must_reverse ? -g.Direction(s) : g.Direction(s);
    const float angle = atan2(Cross2(turn.Back, d), dot(turn.Back, d));
    float clockwise = angle > 0.f ? 2.f * M_PI_F - angle : -angle;
    if (clockwise == 0.f) clockwise = 2.f * M_PI_F;
    if (clockwise < turn.Best) {
        turn.Best = clockwise;
        turn.Piece = p;
    }
}
// Each kept piece links to the kept piece after it, and starts its loop label at its own index and its jump at its link.
kernel void RegionNext(
    uint i [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant JobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const RegionContext ctx{b, pc};
    const RegionJob job = ctx.Job();
    if (i >= job.KeptPieces) return;
    const RegionGeometry g = Geometry(ctx, job);
    const uint p = ctx.Run<uint>(job.Kept)[i];
    const RegionTravel travel = TravelOf(ctx, job, p);
    const RegionSegment segment = g.Segments[travel.Segment];
    const float2 d = travel.Reversed ? -g.Direction(travel.Segment) : g.Direction(travel.Segment);
    RegionTurn turn{-d, INFINITY, InvalidOffset};
    device const uint *crossing_first = ctx.Run<uint>(job.CrossingFirst);
    const bool at_end = !travel.Reversed;
    const bool at_vertex = at_end ? travel.Index == travel.Count : travel.Index == 0u;
    if (at_vertex) {
        const uint v = at_end ? segment.End : segment.Start;
        device const uint *first = ctx.Run<uint>(job.IncidentFirst), *incident = ctx.Run<uint>(job.Incident);
        for (uint j = first[v]; j < first[v + 1u]; ++j) {
            const uint l = incident[j] & ~RegionIncidentEnd;
            const bool end = (incident[j] & RegionIncidentEnd) != 0u;
            const uint count = crossing_first[l + 1u] - crossing_first[l];
            Consider(ctx, job, g, turn, l, end ? count : 0u, end);
        }
    } else {
        const uint k = at_end ? travel.Index : travel.Index - 1u;
        const uint e = crossing_first[travel.Segment] + k;
        const uint t = ctx.Run<RegionCrossing>(job.Crossings)[e].Other;
        uint j = crossing_first[t];
        while (j < crossing_first[t + 1u] && ctx.Run<RegionCrossing>(job.Crossings)[j].Other != travel.Segment) ++j;
        const uint kt = j - crossing_first[t];
        Consider(ctx, job, g, turn, travel.Segment, k + 1u, false);
        Consider(ctx, job, g, turn, travel.Segment, k, true);
        if (j < crossing_first[t + 1u]) {
            Consider(ctx, job, g, turn, t, kt + 1u, false);
            Consider(ctx, job, g, turn, t, kt, true);
        }
    }
    // A piece with no continuation marks the arrangement degenerate and loops to itself, so the rank passes stay in bounds.
    if (turn.Piece == InvalidOffset) ctx.Run<uint>(job.Fault)[0] = 1u;
    const uint next = turn.Piece == InvalidOffset ? i : ctx.Run<uint>(job.PieceKept)[turn.Piece];
    ctx.Run<uint>(job.Next)[i] = next;
    ctx.Run<uint>(job.Labels)[i] = i;
    ctx.Run<uint>(job.Jumps)[i] = next;
}

kernel void RegionLabel(
    uint i [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant JobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const RegionContext ctx{b, pc};
    const RegionJob job = ctx.Job();
    if (i < job.KeptPieces) LabelRound(ctx.Run<uint>(job.Labels), ctx.Run<uint>(job.Jumps), job.KeptPieces, pc.Parameter & 1u, i);
}
// Each loop breaks before its representative, the piece labeled with its own index.
// List ranking counts the pieces from each to the break, so the representative holds the loop's length.
kernel void RegionRankInit(
    uint i [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant JobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const RegionContext ctx{b, pc};
    const RegionJob job = ctx.Job();
    if (i >= job.KeptPieces) return;
    const uint label = ctx.Run<uint>(job.Labels)[pc.Parameter * job.KeptPieces + i];
    ctx.Run<uint>(job.Distances)[i] = 1u;
    const uint next = ctx.Run<uint>(job.Next)[i];
    ctx.Run<uint>(job.Successors)[i] = next == label ? InvalidOffset : next;
}
kernel void RegionRank(
    uint i [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant JobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const RegionContext ctx{b, pc};
    const RegionJob job = ctx.Job();
    if (i < job.KeptPieces) RankRound(ctx.Run<uint>(job.Distances), ctx.Run<uint>(job.Successors), job.KeptPieces, pc.Parameter & 1u, i);
}
// Each representative counts its loop and its length for the scans, with a zero terminator.
// Parameter holds the final buffers' parity.
kernel void RegionLoopCount(
    uint i [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant JobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const RegionContext ctx{b, pc};
    const RegionJob job = ctx.Job();
    if (i > job.KeptPieces) return;
    device uint *lengths = ctx.Run<uint>(job.LoopLengths), *flags = ctx.Run<uint>(job.LoopFlags);
    if (i == job.KeptPieces) {
        lengths[i] = flags[i] = 0u;
        return;
    }
    const uint converged = pc.Parameter * job.KeptPieces;
    const bool representative = ctx.Run<uint>(job.Labels)[converged + i] == i;
    lengths[i] = representative ? ctx.Run<uint>(job.Distances)[converged + i] : 0u;
    flags[i] = representative ? 1u : 0u;
}
kernel void RegionOutput(
    uint i [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant JobPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const RegionContext ctx{b, pc};
    const RegionJob job = ctx.Job();
    if (i >= job.KeptPieces) return;
    const RegionGeometry g = Geometry(ctx, job);
    const uint converged = pc.Parameter * job.KeptPieces;
    const uint label = ctx.Run<uint>(job.Labels)[converged + i];
    device const uint *distances = ctx.Run<uint>(job.Distances) + converged;
    device const uint *lengths = ctx.Run<uint>(job.LoopLengths);
    const uint offset = lengths[label], length = distances[label];
    if (label == i) ctx.Run<packed_uint2>(job.LoopRecords)[ctx.Run<uint>(job.LoopFlags)[i]] = packed_uint2(offset, length);
    const RegionTravel travel = TravelOf(ctx, job, ctx.Run<uint>(job.Kept)[i]);
    ctx.Run<packed_float2>(job.Output)[offset + length - distances[i]] = packed_float2(PiecePoint(ctx, job, g, travel.Segment, travel.Index, travel.Reversed));
}
