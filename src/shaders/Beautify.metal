#include "ConnectivityRead.metal"
#include "gpu/BeautifyPushConstants.h"

constant float BeautifyNoCost = FLT_MAX;

struct BeautifyContext {
    device const BindlessSet &B;
    constant BeautifyPushConstants &Pc;
    ConnectivityView View;
    BeautifyContext(device const BindlessSet &b, constant BeautifyPushConstants &pc) : B(b), Pc(pc), View{b, pc.Connectivity} {}
    device uint *Run(uint offset) const { return BindlessBufferMutable(uint, B.Buffer, Pc.ScratchSlot) + offset; }
    device uint *Node(uint n) const { return Run(Pc.Nodes) + BeautifyNodeWords * n; }
    device uint *Edge(uint e) const { return Run(Pc.Edges) + BeautifyEdgeWords * e; }
    KeyTable Table() const { return {Run(Pc.Table), Pc.TableMask}; }
    uint Vertex(uint n) const { return Node(n)[1]; }
    float3 Position(uint n) const { return float3(BindlessBuffer(struct Vertex, B.VertexBuffer, Pc.VertexSlot)[View.VertexAddress(Vertex(n))].Position); }
    // True for a faced mesh edge that the soup did not own at the start.
    bool MeshExists(uint2 key) const {
        const uint e = View.EdgeBetween(key.x, key.y);
        return e != InvalidOffset && View.EdgeSides(e) != 0u && Run(Pc.SoupOwned)[e] == 0u;
    }
    bool Selected(uint e) const { return SortedContains(Run(Pc.SelectedEdges), Pc.SelectedEdgeCount, e); }
    static uint Next(uint n) { return n - n % 3u + (n + 1u) % 3u; }
    static uint Prev(uint n) { return n - n % 3u + (n + 2u) % 3u; }
    static uint2 Key(uint a, uint b) { return uint2(min(a, b), max(a, b)); }
    uint2 KeyOf(uint n) const { return Key(Vertex(n), Vertex(Next(n))); }
    // The key of the diagonal a soup edge rotates to, between the corners before its two users.
    uint2 Diagonal(device const uint *edge) const { return Key(Vertex(Prev(edge[2])), Vertex(Prev(edge[3]))); }
    // True when a slot written before this pass holds the key.
    bool TableExists(uint2 key) const { return Find(key) != nullptr; }
    void Claim(uint2 key) const { Table().Claim(key, HashPair(key.x, key.y)); }
    // The first slot holding the key, once every claim of the round has written its value.
    device uint *Find(uint2 key) const { return Table().Find(key, HashPair(key.x, key.y)); }
};

inline float3 BeautifyCross(float3 a, float3 b, float3 c) { return cross(b - a, c - a); }
// Blender's BLI_polyfill_edge_calc_rotate_beauty__area over the quad a, b, c, d, whose diagonal b-d rotates to a-c.
// The cost is negative where the rotation raises the two triangles' summed area-to-perimeter ratios.
inline float BeautifyCost(float3 a, float3 b, float3 c, float3 d) {
    const float3 sum = BeautifyCross(b, c, d) + BeautifyCross(b, d, a);
    const float scale = length(sum);
    if (!(scale > 0.f)) return BeautifyNoCost;
    const float3 normal = sum / scale;
    b -= a;
    c -= a;
    d -= a;
    a = float3(0.f);
    b -= normal * dot(b, normal);
    c -= normal * dot(c, normal);
    d -= normal * dot(d, normal);
    const float old_a = dot(BeautifyCross(b, c, d), normal), old_b = dot(BeautifyCross(b, d, a), normal);
    const float new_a = dot(BeautifyCross(a, b, c), normal), new_b = dot(BeautifyCross(a, c, d), normal);
    const auto sign = [](float x) { return x > 1e-5f ? 1 : x < -1e-5f ? -1 : 0; };
    if (!(sign(old_a / scale) + sign(old_b / scale))) return BeautifyNoCost;
    if ((new_a >= 0.f) != (new_b >= 0.f) || abs(new_a) <= 1e-12f || abs(new_b) <= 1e-12f || (old_a >= 0.f) != (old_b >= 0.f)) return BeautifyNoCost;
    if (abs(old_a) <= 1e-12f || abs(old_b) <= 1e-12f) return -BeautifyNoCost;
    const float ab = length(a - b), bc = length(b - c), cd = length(c - d), da = length(d - a), ac = length(a - c), bd = length(b - d);
    const float old_quality = abs(old_a) / (bc + cd + bd) + abs(old_b) / (da + ab + bd);
    const float new_quality = abs(new_a) / (ab + bc + ac) + abs(new_b) / (cd + da + ac);
    // A cost within rounding of zero counts as zero, so equal-quality diagonals do not rotate.
    const float cost = old_quality - new_quality;
    return abs(cost) <= 8.f * FLT_EPSILON * max(old_quality, new_quality) ? 0.f : cost;
}

// Each corner of a selected triangle records its halfedge and vertex and claims a slot for the edge key it starts.
// A corner of any other face writes InvalidOffset.
kernel void BeautifyCorners(uint n [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant BeautifyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    if (n >= pc.NodeCount) return;
    const BeautifyContext ctx{b, pc};
    device uint *node = ctx.Node(n);
    const uint f = ctx.Run(pc.Faces)[n / 3u];
    const uint2 range = ctx.View.FaceHalfedges(f);
    if (range.y - range.x != 3u) {
        node[0] = node[1] = node[2] = InvalidOffset;
        return;
    }
    const uint h = range.x + n % 3u;
    const uint a = ctx.View.VertexOf(h);
    node[0] = h;
    node[1] = a;
    const uint next = range.x + (n + 1u) % 3u;
    ctx.Claim(ctx.Key(a, ctx.View.VertexOf(next)));
}
// Each corner finds its key's first slot and offers itself as its owner, the lowest corner winning.
kernel void BeautifyOwners(uint n [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant BeautifyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    if (n >= pc.NodeCount) return;
    const BeautifyContext ctx{b, pc};
    if (ctx.Node(n)[0] == InvalidOffset) return;
    atomic_fetch_min_explicit(AtomicWord(ctx.Find(ctx.KeyOf(n)) + 2u), n, memory_order_relaxed);
}
// Each corner joins its owner's users and marks the owner, which the scan turns into compact edge indices.
kernel void BeautifyUsers(uint n [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant BeautifyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    if (n > pc.NodeCount) return;
    const BeautifyContext ctx{b, pc};
    device uint *owners = ctx.Run(pc.Owners);
    owners[n] = 0u;
    if (n == pc.NodeCount || ctx.Node(n)[0] == InvalidOffset) return;
    const uint owner = ctx.Find(ctx.KeyOf(n))[2];
    ctx.Node(n)[2] = owner;
    atomic_fetch_add_explicit(AtomicWord(ctx.Run(pc.Users) + owner), 1u, memory_order_relaxed);
    atomic_fetch_max_explicit(AtomicWord(ctx.Run(pc.Second) + owner), n, memory_order_relaxed);
    if (owner == n) owners[n] = 1u;
}
// An eligible edge is selected and has exactly two faces, both in the soup.
// An edge whose faces are all in the soup marks its mesh edge as soup-owned.
kernel void BeautifyEdges(uint n [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant BeautifyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    if (n >= pc.NodeCount) return;
    const BeautifyContext ctx{b, pc};
    device const uint *owners = ctx.Run(pc.Owners);
    if (owners[n + 1u] == owners[n]) return;
    device uint *edge = ctx.Edge(owners[n]);
    const uint users = ctx.Run(pc.Users)[n], source = ctx.Node(ctx.Next(n))[0], mesh_edge = ctx.View.Edge(source), sides = ctx.View.EdgeSides(mesh_edge);
    edge[0] = source;
    edge[1] = sides == 2u && users == 2u && ctx.Selected(mesh_edge) ? 1u : 0u;
    edge[2] = n;
    edge[3] = users >= 2u ? ctx.Run(pc.Second)[n] : InvalidOffset;
    if (sides <= users) ctx.Run(pc.SoupOwned)[mesh_edge] = 1u;
}
kernel void BeautifyNodes(uint n [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant BeautifyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    if (n >= pc.NodeCount) return;
    const BeautifyContext ctx{b, pc};
    device uint *node = ctx.Node(n);
    if (node[0] == InvalidOffset) return;
    node[2] = ctx.Run(pc.Owners)[node[2]];
}

kernel void BeautifyInsertCurrent(uint e [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant BeautifyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    if (e >= pc.EdgeCount) return;
    const BeautifyContext ctx{b, pc};
    ctx.Claim(ctx.KeyOf(ctx.Edge(e)[2]));
}
kernel void BeautifyCost(uint e [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant BeautifyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    if (e >= pc.EdgeCount) return;
    const BeautifyContext ctx{b, pc};
    device uint *edge = ctx.Edge(e);
    ctx.Run(pc.Flags)[e] = 0u;
    if (!edge[1]) return;
    const uint h = edge[2], g = edge[3];
    if (ctx.Vertex(ctx.Prev(h)) == ctx.Vertex(ctx.Prev(g))) return;
    const float cost = BeautifyCost(ctx.Position(ctx.Prev(h)), ctx.Position(h), ctx.Position(ctx.Prev(g)), ctx.Position(ctx.Next(h)));
    edge[4] = as_type<uint>(cost);
    if (cost < 0.f) ctx.Run(pc.Flags)[e] = BeautifyFlagCandidate;
}
// A candidate rotates when no candidate on its two triangles costs less, or the same with a lower index.
// Its new diagonal must not already be an edge, and it claims the diagonal for the round.
kernel void BeautifySelect(uint e [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant BeautifyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    if (e >= pc.EdgeCount) return;
    const BeautifyContext ctx{b, pc};
    device const uint *flags = ctx.Run(pc.Flags);
    if (flags[e] != BeautifyFlagCandidate) return;
    device const uint *edge = ctx.Edge(e);
    const float cost = as_type<float>(edge[4]);
    for (uint side = 0u; side < 2u; ++side) {
        const uint corner = edge[2u + side];
        for (uint k = 0u; k < 2u; ++k) {
            const uint o = ctx.Node(k ? ctx.Prev(corner) : ctx.Next(corner))[2];
            if (o == e || flags[o] != BeautifyFlagCandidate) continue;
            const float other = as_type<float>(ctx.Edge(o)[4]);
            if (other < cost || (other == cost && o < e)) return;
        }
    }
    const uint2 key = ctx.Diagonal(edge);
    if (ctx.MeshExists(key) || ctx.TableExists(key)) return;
    ctx.Claim(key);
    ctx.Run(pc.Flags)[e] = BeautifyFlagSelected;
}
// Selected edges creating the same diagonal keep the lowest.
kernel void BeautifyResolve(uint e [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant BeautifyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    if (e >= pc.EdgeCount) return;
    const BeautifyContext ctx{b, pc};
    if (ctx.Run(pc.Flags)[e] != BeautifyFlagSelected) return;
    device const uint *edge = ctx.Edge(e);
    atomic_fetch_min_explicit(AtomicWord(ctx.Find(ctx.Diagonal(edge)) + 2u), e, memory_order_relaxed);
}
// Blender's BM_edge_rotate over the soup: the two triangles take the other diagonal, and each outer edge keeps its identity under its new corner.
kernel void BeautifyFlip(uint e [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant BeautifyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    if (e >= pc.EdgeCount) return;
    const BeautifyContext ctx{b, pc};
    if (ctx.Run(pc.Flags)[e] != BeautifyFlagSelected) return;
    device uint *edge = ctx.Edge(e);
    const uint h = edge[2], g = edge[3];
    if (ctx.Find(ctx.Diagonal(edge))[2] != e) return;
    const uint hb = ctx.Next(h), hc = ctx.Prev(h), ga = ctx.Next(g), gd = ctx.Prev(g);
    const bool same = ctx.Vertex(h) == ctx.Vertex(g);
    const uint2 a = uint2(ctx.Node(same ? g : ga)[0], ctx.Vertex(same ? g : ga)), bb = uint2(ctx.Node(hb)[0], ctx.Vertex(hb)), c = uint2(ctx.Node(hc)[0], ctx.Vertex(hc)), d = uint2(ctx.Node(gd)[0], ctx.Vertex(gd));
    const uint e_hb = ctx.Node(hb)[2], e_hc = ctx.Node(hc)[2], e_ga = ctx.Node(ga)[2], e_gd = ctx.Node(gd)[2];
    const auto place = [&](uint n, uint2 source, uint to) {
        ctx.Node(n)[0] = source.x;
        ctx.Node(n)[1] = source.y;
        ctx.Node(n)[2] = to;
    };
    const auto replace = [&](uint at, uint from, uint to) {
        device uint *users = ctx.Edge(at) + 2u;
        if (users[0] == from) users[0] = to;
        else if (users[1] == from) users[1] = to;
    };
    place(h, c, e);
    place(hb, d, same ? e_ga : e_gd);
    place(hc, bb, e_hb);
    place(g, d, same ? e_gd : e);
    place(ga, same ? a : c, e_hc);
    place(gd, same ? c : a, same ? e : e_ga);
    replace(e_hb, hb, hc);
    replace(e_hc, hc, ga);
    if (same) {
        replace(e_ga, ga, hb);
        replace(e_gd, gd, g);
        edge[3] = gd;
    } else {
        replace(e_gd, gd, hb);
        replace(e_ga, ga, gd);
    }
    ctx.Run(pc.Changed)[0] = 1u;
    atomic_fetch_add_explicit(AtomicWord(ctx.Run(pc.Rotations)), 1u, memory_order_relaxed);
}
