// Edge collapses over a record's connectivity, elected in independent sets.
#include "ConnectivityRead.metal"
#include "SimplifyQuadric.metal"
#include "gpu/MeshAttributeBit.h"
#include "gpu/Simplify.h"

// A job table pass's job of threadgroup `group`, its record's SimplifyJob, and the job's thread `I` at `lane`.
struct SimplifyAt {
    device const BindlessSet &B;
    uint I;
    constant SimplifyPass &Pass;
    device SimplifyJob &Job;
    SimplifyAt(device const BindlessSet &b, constant JobTablePushConstants &table, constant SimplifyPass *jobs, uint group, uint lane)
        : B(b), Pass(TileJobOf(jobs, b, table, group, I)), Job(*reinterpret_cast<device SimplifyJob *>(BindlessBufferMutable(uint, b.Buffer, Pass.Job.Slot) + Pass.Job.Offset)) { I = I * JobTileThreads + lane; }
    device uint *Words(uint offset) const { return BindlessBufferMutable(uint, B.Buffer, Pass.Job.Slot) + offset; }
    // The collapses the election pass of the parity elected.
    device uint &Elected(uint parity) const { return Words(Job.Counts)[SimplifyElected + (parity & 1u)]; }
    // Whether the trim runs: the election converged with collapses whose reduction exceeds the budget.
    bool Trims() const {
        device const uint *counts = Words(Job.Counts);
        return Elected((Pass.Parameter & SimplifyLastParity) != 0u) == 0u && counts[0] && counts[1] > Job.Budget;
    }
};
// A pass's view of its record, holding the record's references the pass reads per element.
struct SimplifyMesh {
    device const BindlessSet &B;
    device SimplifyJob &Job;
    uint Slot, VertexSlot, FaceNormalSlot, QuadricRun, FlagRun;
    SlotOffset SelectedBits;
    ConnectivityView Conn;
    SimplifyMesh(const thread SimplifyAt &at)
        : B(at.B), Job(at.Job), Slot(at.Pass.Job.Slot), VertexSlot(at.Job.Arenas.VertexSlot), FaceNormalSlot(at.Job.Arenas.BaseFaceNormalSlot), QuadricRun(at.Job.Quadrics), FlagRun(at.Job.Flags), SelectedBits(at.Job.Selected), Conn(at.B, at.Job.Connectivity) {}

    device uint *Words(uint offset) const { return BindlessBufferMutable(uint, B.Buffer, Slot) + offset; }
    device float *Floats(uint offset) const { return reinterpret_cast<device float *>(Words(offset)); }
    device float *Quadric(uint v) const { return Floats(QuadricRun) + SimplifyQuadricWords * v; }
    float3 Position(uint v) const { return float3(BindlessBuffer(Vertex, B.VertexBuffer, VertexSlot)[Conn.VertexAddress(v)].Position); }
    float3 FaceNormal(uint f) const { return float3(BindlessBuffer(packed_float3, B.Buffer, FaceNormalSlot)[Conn.FaceAddress(f)]); }
    // An edge's ends, lower handle first.
    uint2 Ends(uint e) const {
        const uint2 ends = Conn.EdgeEnds(e);
        return uint2(min(ends.x, ends.y), max(ends.x, ends.y));
    }
    bool Selected(uint v) const { return MaskBit(BindlessBuffer(uint, B.Buffer, SelectedBits.Slot) + SelectedBits.Offset, v); }
    uint Flags(uint v) const { return Words(FlagRun)[v]; }
    bool Adjacent(uint v, uint n) const { return Conn.FaceSides(v, n) != 0u; }
    template<typename Visit> void ForAffected(uint a, uint b, Visit visit) const {
        for (const auto item : Conn.Fan(a)) {
            if (item.y != InvalidOffset) visit(item.y);
        }
        for (const auto item : Conn.Fan(b)) {
            if (item.y != InvalidOffset && !Conn.FaceHas(item.y, a)) visit(item.y);
        }
    }
    // Calls `visit(k, vertex)` over the face's loop once `b` joins `a`, dropping consecutive repeats and a last vertex repeating the first.
    // Returns the loop's vertex count.
    template<typename Visit> uint Replaced(uint f, uint a, uint b, Visit visit) const {
        const uint2 range = Conn.FaceHalfedges(f);
        uint count = 0u, first = InvalidOffset, last = InvalidOffset;
        for (uint h = range.x; h < range.y; ++h) {
            uint v = Conn.VertexOf(h);
            if (v == b) v = a;
            if (count && v == last) continue;
            if (first == InvalidOffset) first = v;
            last = v;
            ++count;
        }
        if (count > 1u && first == last) --count;
        uint k = 0u;
        last = InvalidOffset;
        for (uint h = range.x; h < range.y && k < count; ++h) {
            uint v = Conn.VertexOf(h);
            if (v == b) v = a;
            if (k && v == last) continue;
            last = v;
            visit(k++, v);
        }
        return count;
    }
    uint ReplacedAt(uint f, uint a, uint b, uint at) const {
        uint found = InvalidOffset;
        Replaced(f, a, b, [&](uint k, uint v) { if (k == at) found = v; });
        return found;
    }
    // The area vector of a loop, with `a` at `position` once `b` joins it, when `replaced`.
    float3 Normal(uint f, uint a, uint b, bool replaced, float3 position) const {
        float3 n = float3(0.f), origin = float3(0.f), previous = float3(0.f);
        uint count = 0u;
        const auto accumulate = [&](uint k, uint v) {
            const float3 p = replaced && v == a ? position : Position(v);
            if (k == 0u) origin = p;
            else if (k >= 2u) n += cross(previous - origin, p - origin);
            previous = p;
            ++count;
        };
        if (replaced) Replaced(f, a, b, accumulate);
        else {
            const uint2 range = Conn.FaceHalfedges(f);
            for (uint h = range.x; h < range.y; ++h) accumulate(h - range.x, Conn.VertexOf(h));
        }
        return n;
    }
};

// A boundary vertex adds the planes through its boundary edges along their faces' normals, weighted a hundredfold.
kernel void SimplifyVertexSetup(JOB_TABLE_ARGS(SimplifyPass)) {
    const SimplifyAt at{b, table, jobs, group, lane};
    const uint v = BlockHandle(at.I, b, at.Job.Vertices);
    if (v == InvalidOffset) return;
    const SimplifyMesh m{at};
    SimplifyQuadric q = SimplifyQuadric::Zero(m.Position(v));
    uint flags = m.Selected(v) ? SimplifySelected : 0u;
    for (const auto item : m.Conn.Fan(v)) {
        if (item.y == InvalidOffset) continue;
        const float3 n = m.FaceNormal(item.y);
        const float d = -dot(n, m.Position(m.Conn.VertexOf(m.Conn.FaceHalfedges(item.y).x)));
        if (isfinite(d) && length(n) > 0.f) q.Add(n, d, 1.f);
    }
    m.Conn.ForEachIncidentEdge(v, [&](uint e) {
        if (m.Conn.EdgeFaceCount(e) != 1u) return;
        flags |= SimplifyBoundary;
        const uint2 ends = m.Ends(e), faces = m.Conn.EdgeFaces(e);
        const float3 direction = m.Position(ends.y) - m.Position(ends.x);
        const float3 normal = cross(direction, m.FaceNormal(faces.x != InvalidOffset ? faces.x : faces.y));
        const float length = metal::length(normal);
        if (length > 0.f) q.Add(normal / length, -dot(normal / length, m.Position(ends.x)), 100.f);
    });
    q.Store(m.Quadric(v));
    m.Words(m.FlagRun)[v] = flags;
}
// An edge locks its ends where it is sharp, where its faces' primitives differ, or where a corner layer differs across it at either end.
kernel void SimplifyEdgeLocks(JOB_TABLE_ARGS(SimplifyPass)) {
    const SimplifyAt at{b, table, jobs, group, lane};
    const uint e = BlockHandle(at.I, b, at.Job.Edges);
    if (e == InvalidOffset) return;
    const SimplifyMesh m{at};
    device const SimplifyJob &pc = at.Job;
    bool locked = BindlessBuffer(uchar, b.Buffer, pc.Arenas.EdgeSharpnessSlot)[m.Conn.EdgeAddress(e)] != 0u;
    const uint h = m.Conn.EdgeHalfedge(e), o = m.Conn.Opposite(h);
    const uint2 faces = m.Conn.EdgeFaces(e), ends = m.Ends(e);
    locked = locked || m.Conn.FaceSides(ends.x, ends.y) > 2u;
    if (faces.x != InvalidOffset && faces.y != InvalidOffset) {
        if (pc.Layers.FacePrimitives.Slot != InvalidSlot) {
            device const uint *primitives = BindlessBuffer(uint, b.ElementPrimitiveBuffer, pc.Layers.FacePrimitives.Slot) + pc.Layers.FacePrimitives.Offset;
            locked = locked || primitives[faces.x] != primitives[faces.y];
        }
        // The corners at the arriving end are h and the one before o, and at the leaving end the one before h and o.
        const uint2 pairs[2] = {uint2(h, m.Conn.Previous(o)), uint2(m.Conn.Previous(h), o)};
        for (uint side = 0u; side < 2u; ++side) {
            const uint c = pairs[side].x, k = pairs[side].y;
            for (uint set = 0u; set < 4u; ++set) {
                if (!(pc.CornerAttributes & (MeshAttributeBit_TexCoord0 << set))) continue;
                device const packed_float2 *uvs = BindlessBuffer(packed_float2, b.CornerUvBuffer, pc.Layers.CornerUvs[set].Slot) + pc.Layers.CornerUvs[set].Offset;
                locked = locked || any(float2(uvs[c]) != float2(uvs[k]));
            }
            if (pc.CornerAttributes & MeshAttributeBit_Color0) {
                device const packed_float4 *colors = BindlessBuffer(packed_float4, b.CornerColorBuffer, pc.Layers.CornerColor.Slot) + pc.Layers.CornerColor.Offset;
                locked = locked || any(float4(colors[c]) != float4(colors[k]));
            }
            if (pc.CornerAttributes & MeshAttributeBit_Tangent) {
                device const packed_float4 *tangents = BindlessBuffer(packed_float4, b.CornerTangentBuffer, pc.Layers.CornerTangent.Slot) + pc.Layers.CornerTangent.Offset;
                locked = locked || any(float4(tangents[c]) != float4(tangents[k]));
            }
        }
    }
    if (!locked) return;
    device atomic_uint *flags = AtomicWord(m.Words(m.FlagRun));
    atomic_fetch_or_explicit(&flags[ends.x], SimplifyLocked, memory_order_relaxed);
    atomic_fetch_or_explicit(&flags[ends.y], SimplifyLocked, memory_order_relaxed);
}
kernel void SimplifyFaceTriangles(JOB_TABLE_ARGS(SimplifyPass)) {
    const SimplifyAt at{b, table, jobs, group, lane};
    const uint f = BlockHandle(at.I, b, at.Job.Faces);
    if (f == InvalidOffset) return;
    const SimplifyMesh m{at};
    const uint2 range = m.Conn.FaceHalfedges(f);
    bool selected = false;
    for (uint h = range.x; h < range.y; ++h) selected = selected || m.Selected(m.Conn.VertexOf(h));
    if (!selected || range.y - range.x < 3u) return;
    atomic_fetch_add_explicit(AtomicWord(m.Words(m.Job.Counts)) + 2u, range.y - range.x - 2u, memory_order_relaxed);
}

// Whether collapsing `b` onto `a` at `position` keeps the faces around them simple, unflipped and distinct.
inline bool SimplifyValid(SimplifyMesh m, uint a, uint b, float3 position) {
    uint users = 0u;
    for (const auto item : m.Conn.Fan(a)) {
        const uint f = item.y;
        if (f == InvalidOffset || !m.Conn.FaceHas(f, b)) continue;
        const uint2 range = m.Conn.FaceHalfedges(f);
        for (uint h = range.x; h < range.y; ++h) {
            const uint v = m.Conn.VertexOf(h), n = m.Conn.VertexOf(m.Conn.Next(h, f));
            if ((v == a && n == b) || (v == b && n == a)) ++users;
        }
    }
    if (users < 1u || users > 2u) return false;
    for (const auto item : m.Conn.Fan(a)) {
        if (item.y == InvalidOffset) continue;
        const uint neighbors[2] = {m.Conn.VertexOf(m.Conn.Next(item.x, item.y)), m.Conn.From(item.x)};
        for (uint side = 0u; side < 2u; ++side) {
            const uint c = neighbors[side];
            if (c == a || c == b || !m.Adjacent(b, c)) continue;
            bool opposite = false;
            for (const auto shared : m.Conn.Fan(a)) {
                const uint f = shared.y;
                if (f == InvalidOffset) continue;
                opposite = opposite || (m.Conn.Valence(f) == 3u && m.Conn.FaceHas(f, b) && m.Conn.FaceHas(f, c));
            }
            if (!opposite) return false;
        }
    }
    const bool boundary_a = (m.Flags(a) & SimplifyBoundary) != 0u, boundary_b = (m.Flags(b) & SimplifyBoundary) != 0u;
    if (boundary_a != boundary_b || (boundary_a && users != 1u)) return false;
    uint kept = 0u;
    bool valid = true;
    m.ForAffected(a, b, [&](uint f) {
        if (!valid) return;
        const uint count = m.Replaced(f, a, b, [](uint, uint) {});
        if (count < 3u) return;
        const float3 before = m.Normal(f, a, b, false, position), after = m.Normal(f, a, b, true, position);
        if (!(dot(before, after) > 1e-6f * length(before) * length(after)) || !(length(after) > 1e-8f * length(before))) {
            valid = false;
            return;
        }
        for (uint i = 0u; i < count && valid; ++i) {
            const uint v = m.ReplacedAt(f, a, b, i);
            for (uint j = i + 1u; j < count; ++j) valid = valid && m.ReplacedAt(f, a, b, j) != v;
        }
        uint x = InvalidOffset;
        m.Replaced(f, a, b, [&](uint, uint v) { if (x == InvalidOffset && v != a) x = v; });
        for (const auto item : m.Conn.Fan(x)) {
            const uint g = item.y;
            if (!valid || g == InvalidOffset || g <= f || !(m.Conn.FaceHas(g, a) || m.Conn.FaceHas(g, b)) || m.Replaced(g, a, b, [](uint, uint) {}) != count) continue;
            bool same = true;
            for (uint i = 0u; i < count && same; ++i) {
                const uint v = m.ReplacedAt(f, a, b, i);
                bool held = false;
                for (uint j = 0u; j < count; ++j) held = held || m.ReplacedAt(g, a, b, j) == v;
                same = held;
            }
            if (same) valid = false;
        }
        ++kept;
    });
    return valid && kept > 0u;
}

// Each edge between selected unlocked ends proposes collapsing its higher end onto its lower, placed at the cheaper end, with its cost.
kernel void SimplifyCandidates(JOB_TABLE_ARGS(SimplifyPass)) {
    const SimplifyAt at{b, table, jobs, group, lane};
    const uint e = BlockHandle(at.I, b, at.Job.Edges);
    if (e == InvalidOffset) return;
    const SimplifyMesh m{at};
    device float *errors = m.Floats(m.Job.Errors);
    errors[e] = INFINITY;
    m.Words(m.Job.Winners)[e] = 0u;
    const uint2 ends = m.Ends(e);
    const uint a = ends.x, c = ends.y;
    if (a == c) return;
    const uint flags_a = m.Flags(a), flags_c = m.Flags(c);
    if (!(flags_a & SimplifySelected) || !(flags_c & SimplifySelected) || (flags_a & SimplifyLocked) || (flags_c & SimplifyLocked)) return;
    SimplifyQuadric q = SimplifyQuadric::Load(m.Quadric(a));
    q.Add(SimplifyQuadric::Load(m.Quadric(c)));
    const float3 at_a = m.Position(a), at_c = m.Position(c);
    const float3 p = q.Cost(at_c) < q.Cost(at_a) ? at_c : at_a;
    const float cost = q.Cost(p);
    if (!(cost < m.Job.MaxCost) || !SimplifyValid(m, a, c, p)) return;
    uint reduction = 0u;
    m.ForAffected(a, c, [&](uint f) {
        const uint n = m.Conn.Valence(f), count = m.Replaced(f, a, c, [](uint, uint) {});
        reduction += (n - 2u) - (count >= 3u ? count - 2u : 0u);
    });
    errors[e] = cost;
    device float *positions = m.Floats(m.Job.Positions) + 3u * e;
    positions[0] = p.x;
    positions[1] = p.y;
    positions[2] = p.z;
    m.Words(m.Job.Reductions)[e] = reduction;
}
// A proposal's election priority is its cost's exponent and four leading mantissa bits, then a hash of its ends.
inline uint2 SimplifyPriority(float cost, uint2 ends) {
    return uint2(cost > 0.f ? as_type<uint>(cost) >> 19u : 0u, HashPair(ends.x, ends.y));
}
// Calls `visit(g)` for each edge other than `e` on a face at either of its ends, until a call returns true, returning whether one did.
template<typename Visit> bool SimplifyAnyNear(SimplifyMesh m, uint e, uint2 ends, Visit visit) {
    for (uint side = 0u; side < 2u; ++side) {
        for (const auto item : m.Conn.Fan(side ? ends.y : ends.x)) {
            if (item.y == InvalidOffset) continue;
            const uint2 range = m.Conn.FaceHalfedges(item.y);
            for (uint h = range.x; h < range.y; ++h) {
                const uint g = m.Conn.Edge(h);
                if (g != e && visit(g)) return true;
            }
        }
    }
    return false;
}
// A cost's bits once costs at most `floor` take it.
inline uint SimplifyFloored(float cost, float floor) { return as_type<uint>(cost > floor ? cost : floor); }

// Each undecided proposal at most the threshold runs when no such proposal on a face at either end has a lower priority, with ties going to the lower ends.
// The passes alternate with SimplifyBlock until one elects nothing, electing the proposals greedily in priority order.
// A pass after the first exits at once after a pass that elected nothing, and counts its collapses at its parity.
kernel void SimplifyElect(JOB_TABLE_ARGS(SimplifyPass)) {
    const SimplifyAt at{b, table, jobs, group, lane};
    const uint pass = at.Pass.Parameter;
    if (pass && !at.Elected(pass - 1u)) return;
    const uint e = BlockHandle(at.I, b, at.Job.Edges);
    if (e == InvalidOffset) return;
    const SimplifyMesh m{at};
    device const float *errors = m.Floats(m.Job.Errors);
    device uint *winners = m.Words(m.Job.Winners);
    const float cost = errors[e], threshold = m.Job.Threshold;
    if (winners[e] || !isfinite(cost) || !(cost <= threshold)) return;
    const uint2 ends = m.Ends(e), priority = SimplifyPriority(cost, ends);
    const bool beaten = SimplifyAnyNear(m, e, ends, [&](uint g) {
        const float other = errors[g];
        if (!isfinite(other) || !(other <= threshold)) return false;
        const uint2 o = m.Ends(g), other_priority = SimplifyPriority(other, o);
        if (other_priority.x != priority.x) return other_priority.x < priority.x;
        if (other_priority.y != priority.y) return other_priority.y < priority.y;
        return o.x != ends.x ? o.x < ends.x : o.y < ends.y;
    });
    if (beaten) return;
    winners[e] = 1u;
    device atomic_uint *counts = AtomicWord(m.Words(m.Job.Counts));
    atomic_fetch_add_explicit(counts, 1u, memory_order_relaxed);
    atomic_fetch_add_explicit(counts + 1u, m.Words(m.Job.Reductions)[e], memory_order_relaxed);
    atomic_fetch_add_explicit(counts + SimplifyElected + (pass & 1u), 1u, memory_order_relaxed);
}
// Blocks the undecided proposals near a winner, after an Elect pass that elected any, and clears the count of the Elect pass that follows.
kernel void SimplifyBlock(JOB_TABLE_ARGS(SimplifyPass)) {
    const SimplifyAt at{b, table, jobs, group, lane};
    const uint pass = at.Pass.Parameter;
    if (at.I == 0u) at.Elected(pass) = 0u;
    if (!at.Elected(pass - 1u)) return;
    const uint e = BlockHandle(at.I, b, at.Job.Edges);
    if (e == InvalidOffset) return;
    const SimplifyMesh m{at};
    device float *errors = m.Floats(m.Job.Errors);
    device const uint *winners = m.Words(m.Job.Winners);
    if (winners[e] || !isfinite(errors[e])) return;
    if (SimplifyAnyNear(m, e, m.Ends(e), [&](uint g) { return winners[g] != 0u; })) errors[e] = INFINITY;
}

// The bits a histogram level's mask covers, and the lowest bit its bins hold.
constant uint SimplifyLevelMasks[3] = {0u, 0xfff80000u, 0xffffff80u};
constant uint SimplifyLevelShifts[3] = {19u, 7u, 0u};
// Whether a select pass runs: a trim's runs when the trim does, and a level after the first when the select has not fallen short.
inline bool SimplifySelects(const thread SimplifyAt &at) {
    const uint parameter = at.Pass.Parameter;
    if ((parameter & SimplifySelectWinners) && !at.Trims()) return false;
    return !(parameter & 3u) || at.Job.Bits != as_type<uint>(INFINITY);
}
// Each proposal, or each winner, whose floored cost bits have the select's bits so far adds its reduction to the bin of its bits at the level.
// A winner's cost is not floored.
kernel void SimplifyHistogram(JOB_TABLE_ARGS(SimplifyPass)) {
    const SimplifyAt at{b, table, jobs, group, lane};
    if (!SimplifySelects(at)) return;
    const uint e = BlockHandle(at.I, b, at.Job.Edges);
    if (e == InvalidOffset) return;
    const uint level = at.Pass.Parameter & 3u;
    const bool winners = (at.Pass.Parameter & SimplifySelectWinners) != 0u;
    const float cost = reinterpret_cast<device const float *>(at.Words(at.Job.Errors))[e];
    if (!isfinite(cost) || (winners && !at.Words(at.Job.Winners)[e])) return;
    const uint bits = SimplifyFloored(cost, winners ? 0.f : at.Job.Floor), mask = SimplifyLevelMasks[level];
    if ((bits & mask) != (level ? at.Job.Bits : 0u)) return;
    device atomic_uint *bins = AtomicWord(at.Words(at.Job.Histogram));
    atomic_fetch_add_explicit(bins + ((bits & ~mask) >> SimplifyLevelShifts[level]), at.Words(at.Job.Reductions)[e], memory_order_relaxed);
}
// One threadgroup per record selects the histogram's least bin at which the reductions below the select's bits so far reach the target, appends the bin to the bits, and clears the bins.
// The target is the budget among the proposals and one past it among the winners, and a select whose total falls short takes infinity's bits, which its later levels keep.
// The proposals' select sets the election's threshold to 1.5 times the crossing and clears its counts, and the winners' sets the trim's kept bits and reduction.
kernel void SimplifySelect(JOB_TABLE_ARGS(SimplifyPass), uint simd_lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]) {
    const SimplifyAt at{b, table, jobs, group, lane};
    if (!SimplifySelects(at)) return;
    constexpr uint Per = SimplifyHistogramBins / JobTileThreads;
    const uint level = at.Pass.Parameter & 3u;
    const bool winners = (at.Pass.Parameter & SimplifySelectWinners) != 0u;
    device SimplifyJob &job = at.Job;
    device uint *bins = at.Words(job.Histogram) + lane * Per;
    uint sum = 0u;
    for (uint k = 0u; k < Per; ++k) sum += bins[k];
    threadgroup uint totals[JobTileThreads / 32u];
    const uint within = simd_prefix_exclusive_sum(sum);
    if (simd_lane == 31u) totals[simd] = within + sum;
    const uint bits = level ? job.Bits : 0u, below = level ? job.Below : 0u, target = winners ? job.Budget + 1u : job.Budget;
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    uint start = below + within;
    for (uint s = 0u; s < simd; ++s) start += totals[s];
    const bool found = start < target && start + sum >= target, short_of = lane == JobTileThreads - 1u && start + sum < target;
    if (found) {
        uint bin = 0u, running = start;
        for (; running + bins[bin] < target; ++bin) running += bins[bin];
        job.Bits = bits | ((lane * Per + bin) << SimplifyLevelShifts[level]);
        job.Below = running;
        job.At = bins[bin];
    } else if (short_of) {
        job.Bits = as_type<uint>(INFINITY);
        job.Below = start + sum;
        job.At = 0u;
    }
    for (uint k = 0u; k < Per; ++k) bins[k] = 0u;
    if (!((found && level == 2u) || short_of)) return;
    if (winners) {
        job.Kept = job.Below ? job.Bits - 1u : job.Bits;
        job.Reduced = job.Below ? job.Below : job.At;
        return;
    }
    job.Crossing = job.Bits;
    job.Threshold = 1.5f * as_type<float>(job.Bits);
    device uint *counts = at.Words(job.Counts);
    counts[0] = counts[1] = counts[SimplifyElected] = 0u;
}
// The trim keeps the winners whose unfloored cost bits are at most the kept bits.
kernel void SimplifyTrim(JOB_TABLE_ARGS(SimplifyPass)) {
    const SimplifyAt at{b, table, jobs, group, lane};
    if (!at.Trims()) return;
    const uint e = BlockHandle(at.I, b, at.Job.Edges);
    if (e == InvalidOffset) return;
    device uint *winners = at.Words(at.Job.Winners);
    if (winners[e] && SimplifyFloored(reinterpret_cast<device const float *>(at.Words(at.Job.Errors))[e], 0.f) > at.Job.Kept) winners[e] = 0u;
}
