// Edge collapses over a record's connectivity, elected in independent sets.
#include "ConnectivityRead.metal"
#include "SimplifyQuadric.metal"
#include "gpu/MeshAttributeBit.h"
#include "gpu/Simplify.h"

struct SimplifyMesh {
    device const BindlessSet &B;
    constant SimplifyPushConstants &Pc;
    ConnectivityView Conn;
    device const SimplifyJob &Job;
    SimplifyMesh(device const BindlessSet &b, constant SimplifyPushConstants &pc) : B(b), Pc(pc), Conn(b, pc.Connectivity), Job(*reinterpret_cast<device const SimplifyJob *>(BindlessBuffer(uint, b.Buffer, pc.Job.Slot) + pc.Job.Offset)) {}

    device uint *Words(uint offset) const { return BindlessBufferMutable(uint, B.Buffer, Pc.Job.Slot) + offset; }
    device float *Floats(uint offset) const { return reinterpret_cast<device float *>(Words(offset)); }
    device float *Quadric(uint v) const { return Floats(Job.Quadrics) + SimplifyQuadricWords * v; }
    float3 Position(uint v) const { return float3(BindlessBuffer(Vertex, B.VertexBuffer, Pc.Arenas.VertexSlot)[Conn.VertexAddress(v)].Position); }
    float3 FaceNormal(uint f) const { return float3(BindlessBuffer(packed_float3, B.Buffer, Pc.Arenas.BaseFaceNormalSlot)[Conn.FaceAddress(f)]); }
    // An edge's ends, lower handle first.
    uint2 Ends(uint e) const {
        const uint2 ends = Conn.EdgeEnds(e);
        return uint2(min(ends.x, ends.y), max(ends.x, ends.y));
    }
    bool Selected(uint v) const { return MaskBit(BindlessBuffer(uint, B.Buffer, Pc.Selected.Slot) + Pc.Selected.Offset, v); }
    uint Flags(uint v) const { return Words(Job.Flags)[v]; }
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
kernel void SimplifyVertexSetup(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant SimplifyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    const uint v = BlockHandle(i, b, pc.Blocks);
    if (v == InvalidOffset) return;
    const SimplifyMesh m{b, pc};
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
    m.Words(m.Job.Flags)[v] = flags;
}
// An edge locks its ends where it is sharp, where its faces' primitives differ, or where a corner layer differs across it at either end.
kernel void SimplifyEdgeLocks(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant SimplifyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    const uint e = BlockHandle(i, b, pc.Blocks);
    if (e == InvalidOffset) return;
    const SimplifyMesh m{b, pc};
    bool locked = BindlessBuffer(uchar, b.Buffer, pc.Arenas.EdgeSharpnessSlot)[m.Conn.EdgeAddress(e)] != 0u;
    const uint h = m.Conn.EdgeHalfedge(e), o = m.Conn.Opposite(h);
    const uint2 faces = m.Conn.EdgeFaces(e), ends = m.Ends(e);
    locked = locked || m.Conn.FaceSides(ends.x, ends.y) > 2u;
    if (faces.x != InvalidOffset && faces.y != InvalidOffset) {
        if (pc.Arenas.Layers.FacePrimitives.Slot != InvalidSlot) {
            device const uint *primitives = BindlessBuffer(uint, b.ElementPrimitiveBuffer, pc.Arenas.Layers.FacePrimitives.Slot) + pc.Arenas.Layers.FacePrimitives.Offset;
            locked = locked || primitives[faces.x] != primitives[faces.y];
        }
        // The corners at the arriving end are h and the one before o, and at the leaving end the one before h and o.
        const uint2 pairs[2] = {uint2(h, m.Conn.Previous(o)), uint2(m.Conn.Previous(h), o)};
        for (uint side = 0u; side < 2u; ++side) {
            const uint c = pairs[side].x, k = pairs[side].y;
            for (uint set = 0u; set < 4u; ++set) {
                if (!(pc.CornerAttributes & (MeshAttributeBit_TexCoord0 << set))) continue;
                device const packed_float2 *uvs = BindlessBuffer(packed_float2, b.CornerUvBuffer, pc.Arenas.Layers.CornerUvs[set].Slot) + pc.Arenas.Layers.CornerUvs[set].Offset;
                locked = locked || any(float2(uvs[c]) != float2(uvs[k]));
            }
            if (pc.CornerAttributes & MeshAttributeBit_Color0) {
                device const packed_float4 *colors = BindlessBuffer(packed_float4, b.CornerColorBuffer, pc.Arenas.Layers.CornerColor.Slot) + pc.Arenas.Layers.CornerColor.Offset;
                locked = locked || any(float4(colors[c]) != float4(colors[k]));
            }
            if (pc.CornerAttributes & MeshAttributeBit_Tangent) {
                device const packed_float4 *tangents = BindlessBuffer(packed_float4, b.CornerTangentBuffer, pc.Arenas.Layers.CornerTangent.Slot) + pc.Arenas.Layers.CornerTangent.Offset;
                locked = locked || any(float4(tangents[c]) != float4(tangents[k]));
            }
        }
    }
    if (!locked) return;
    device atomic_uint *flags = AtomicWord(m.Words(m.Job.Flags));
    atomic_fetch_or_explicit(&flags[ends.x], SimplifyLocked, memory_order_relaxed);
    atomic_fetch_or_explicit(&flags[ends.y], SimplifyLocked, memory_order_relaxed);
}
kernel void SimplifyFaceTriangles(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant SimplifyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    const uint f = BlockHandle(i, b, pc.Blocks);
    if (f == InvalidOffset) return;
    const SimplifyMesh m{b, pc};
    const uint2 range = m.Conn.FaceHalfedges(f);
    bool selected = false;
    for (uint h = range.x; h < range.y; ++h) selected = selected || m.Selected(m.Conn.VertexOf(h));
    if (!selected || range.y - range.x < 3u) return;
    atomic_fetch_add_explicit(AtomicWord(m.Words(m.Job.Counts)) + 3u, range.y - range.x - 2u, memory_order_relaxed);
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
kernel void SimplifyCandidates(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant SimplifyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    const uint e = BlockHandle(i, b, pc.Blocks);
    if (e == InvalidOffset) return;
    const SimplifyMesh m{b, pc};
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
inline uint SimplifyFloored(SimplifyMesh m, float cost) { return as_type<uint>(cost > m.Job.Floor ? cost : m.Job.Floor); }

// Each undecided proposal at most the threshold runs when no such proposal on a face at either end has a lower priority, with ties going to the lower ends.
// The passes alternate with SimplifyBlock until one elects nothing, electing the proposals greedily in priority order.
kernel void SimplifyElect(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant SimplifyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    const uint e = BlockHandle(i, b, pc.Blocks);
    if (e == InvalidOffset) return;
    const SimplifyMesh m{b, pc};
    device const float *errors = m.Floats(m.Job.Errors);
    device uint *winners = m.Words(m.Job.Winners);
    const float cost = errors[e];
    if (winners[e] || !isfinite(cost) || !(cost <= m.Job.Threshold)) return;
    const uint2 ends = m.Ends(e), priority = SimplifyPriority(cost, ends);
    const bool beaten = SimplifyAnyNear(m, e, ends, [&](uint g) {
        const float other = errors[g];
        if (!isfinite(other) || !(other <= m.Job.Threshold)) return false;
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
    atomic_fetch_add_explicit(counts + 2u, 1u, memory_order_relaxed);
}
kernel void SimplifyBlock(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant SimplifyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    const uint e = BlockHandle(i, b, pc.Blocks);
    if (e == InvalidOffset) return;
    const SimplifyMesh m{b, pc};
    device float *errors = m.Floats(m.Job.Errors);
    device const uint *winners = m.Words(m.Job.Winners);
    if (winners[e] || !isfinite(errors[e])) return;
    if (SimplifyAnyNear(m, e, m.Ends(e), [&](uint g) { return winners[g] != 0u; })) errors[e] = INFINITY;
}
// Each proposal, or each winner, whose floored cost bits have the histogram's prefix adds its reduction to the bin of its bits outside the mask.
kernel void SimplifyHistogram(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant SimplifyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    const uint e = BlockHandle(i, b, pc.Blocks);
    if (e == InvalidOffset) return;
    const SimplifyMesh m{b, pc};
    const float cost = m.Floats(m.Job.Errors)[e];
    if (!isfinite(cost) || (m.Job.HistogramWinners && !m.Words(m.Job.Winners)[e])) return;
    const uint bits = SimplifyFloored(m, cost);
    if ((bits & m.Job.HistogramMask) != m.Job.HistogramPrefix) return;
    device atomic_uint *bins = AtomicWord(m.Words(m.Job.Histogram));
    atomic_fetch_add_explicit(bins + ((bits & ~m.Job.HistogramMask) >> m.Job.HistogramShift), m.Words(m.Job.Reductions)[e], memory_order_relaxed);
}
kernel void SimplifyTrim(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant SimplifyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    const uint e = BlockHandle(i, b, pc.Blocks);
    if (e == InvalidOffset) return;
    const SimplifyMesh m{b, pc};
    device uint *winners = m.Words(m.Job.Winners);
    if (winners[e] && SimplifyFloored(m, m.Floats(m.Job.Errors)[e]) > m.Job.Kept) winners[e] = 0u;
}
