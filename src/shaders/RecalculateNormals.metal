#include "ConnectivityRead.metal"
#include "OrderedFloat.metal"
#include "gpu/RecalculateNormalsPushConstants.h"

constant uint NormalVolumeFraction = 36u; // Fixed-point bits below the cube of a component's bounds diagonal

struct NormalRecalcContext {
    device const BindlessSet &B;
    constant RecalculateNormalsPushConstants &Pc;
    ConnectivityView View;
    NormalRecalcContext(device const BindlessSet &b, constant RecalculateNormalsPushConstants &pc) : B(b), Pc(pc), View{b, pc.Connectivity} {}
    device uint *Run(uint offset) const { return BindlessBufferMutable(uint, B.Buffer, Pc.ScratchSlot) + offset; }
    uint Face(uint i) const { return BlockHandle(i, B, Pc.Blocks); }
    uint Halfedge(uint i) const { return BlockHandle(i, B, Pc.HalfedgeBlocks); }
    float3 Position(uint h) const { return float3(BindlessBuffer(Vertex, B.VertexBuffer, Pc.VertexSlot)[View.VertexAddress(View.VertexOf(h))].Position); }
    uint Root(uint face) const { return Run(Pc.Labels)[face] >> 1u; }
    uint Parity(uint face) const { return Run(Pc.Labels)[face] & 1u; }
    device uint *Component(uint root) const { return Run(Pc.Components) + NormalComponentWords * root; }
};

inline void AtomicAdd64(device atomic_uint *low, device atomic_uint *high, long value) {
    const uint low_part = uint(value), high_part = uint(ulong(value) >> 32u);
    const uint before = atomic_fetch_add_explicit(low, low_part, memory_order_relaxed);
    const uint carry = before + low_part < before ? 1u : 0u;
    atomic_fetch_add_explicit(high, high_part + carry, memory_order_relaxed);
}
inline long Read64(device const uint *words) { return long((ulong(words[1]) << 32u) | ulong(words[0])); }

// Each halfedge records the face across its edge when exactly two faces share the edge, with the parity set where that face runs the same way.
kernel void RecalculateNormalNeighbors(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant RecalculateNormalsPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    const NormalRecalcContext ctx{b, pc};
    const uint h = ctx.Halfedge(i);
    if (h == InvalidOffset) return;
    const uint face = ctx.View.HalfedgeFace(h);
    uint neighbor = InvalidOffset;
    if (face != InvalidOffset && ctx.View.FaceSides(ctx.View.From(h), ctx.View.VertexOf(h)) == 2u) {
        const uint other = ctx.View.OtherSide(h, face);
        if (other != InvalidOffset) neighbor = ctx.View.HalfedgeFace(other) << 1u | (ctx.View.VertexOf(other) == ctx.View.VertexOf(h) ? 1u : 0u);
    }
    ctx.Run(pc.Neighbors)[h] = neighbor;
}

kernel void RecalculateNormalLabelInit(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant RecalculateNormalsPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    const NormalRecalcContext ctx{b, pc};
    const uint face = ctx.Face(i);
    if (face == InvalidOffset) return;
    ctx.Run(pc.Labels)[face] = face << 1u;
}
// Each label falls to the least of its own, its label's and its neighbors', carrying the parity along the path taken, until a round changes nothing.
kernel void RecalculateNormalLink(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant RecalculateNormalsPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    const NormalRecalcContext ctx{b, pc};
    const uint face = ctx.Face(i);
    if (face == InvalidOffset) return;
    device uint *labels = ctx.Run(pc.Labels);
    const uint current = labels[face];
    const uint through = labels[current >> 1u];
    uint least = min(current, through ^ (current & 1u));
    const uint2 range = ctx.View.FaceHalfedges(face);
    for (uint h = range.x; h < range.y; ++h) {
        const uint neighbor = ctx.Run(pc.Neighbors)[h];
        if (neighbor == InvalidOffset) continue;
        const uint label = labels[neighbor >> 1u];
        least = min(least, label ^ (neighbor & 1u));
    }
    if (least >> 1u < current >> 1u) {
        labels[face] = least;
        ctx.Run(pc.Changed)[0] = 1u;
    }
}
kernel void RecalculateNormalMarkSeeds(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant RecalculateNormalsPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    if (i >= pc.SelectionCount) return;
    const NormalRecalcContext ctx{b, pc};
    ctx.Component(ctx.Root(ctx.Run(pc.Selection)[i]))[NormalComponentMarked] = 1u;
}
kernel void RecalculateNormalBounds(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant RecalculateNormalsPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    const NormalRecalcContext ctx{b, pc};
    const uint face = ctx.Face(i);
    if (face == InvalidOffset) return;
    device uint *component = ctx.Component(ctx.Root(face));
    if (!component[NormalComponentMarked]) return;
    const uint2 loop = ctx.View.FaceHalfedges(face);
    float3 least = ctx.Position(loop.x), greatest = least;
    for (uint h = loop.x + 1u; h < loop.y; ++h) {
        const float3 p = ctx.Position(h);
        least = min(least, p);
        greatest = max(greatest, p);
    }
    for (uint axis = 0u; axis < 3u; ++axis) {
        atomic_fetch_max_explicit(AtomicWord(component + NormalComponentLeast + axis), ~OrderedBits(least[axis]), memory_order_relaxed);
        atomic_fetch_max_explicit(AtomicWord(component + NormalComponentGreatest + axis), OrderedBits(greatest[axis]), memory_order_relaxed);
    }
}
// Each face of a marked component adds six times the signed volume of its fan triangles with the bounds center, wound as the component's lowest face.
// The sum is fixed point in units of the bounds diagonal cubed, so it is exact in any order.
kernel void RecalculateNormalVolumes(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant RecalculateNormalsPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    const NormalRecalcContext ctx{b, pc};
    const uint face = ctx.Face(i);
    if (face == InvalidOffset) return;
    device uint *component = ctx.Component(ctx.Root(face));
    if (!component[NormalComponentMarked]) return;
    float3 least, greatest;
    for (uint axis = 0u; axis < 3u; ++axis) {
        least[axis] = OrderedFloat(~component[NormalComponentLeast + axis]);
        greatest[axis] = OrderedFloat(component[NormalComponentGreatest + axis]);
    }
    const float3 center = (least + greatest) * .5f;
    const float diagonal = max(distance(least, greatest), FLT_MIN);
    const uint2 loop = ctx.View.FaceHalfedges(face);
    const float3 first = (ctx.Position(loop.x) - center) / diagonal;
    float3 previous = (ctx.Position(loop.x + 1u) - center) / diagonal;
    float volume = 0.f;
    for (uint h = loop.x + 2u; h < loop.y; ++h) {
        const float3 p = (ctx.Position(h) - center) / diagonal;
        volume += dot(first, cross(previous, p));
        previous = p;
    }
    const long units = long(rint(volume * float(1ul << NormalVolumeFraction)));
    AtomicAdd64(AtomicWord(component + NormalComponentVolume), AtomicWord(component + NormalComponentVolume + 1u), ctx.Parity(face) ? -units : units);
    for (uint h = loop.x; h < loop.y; ++h) {
        if (ctx.Run(pc.Neighbors)[h] == InvalidOffset) {
            atomic_store_explicit(AtomicWord(component + NormalComponentOpen), 1u, memory_order_relaxed);
            break;
        }
    }
}
// Each selected face of a closed component flips where its parity disagrees with the sign of the component's volume, and a face of an open component stays.
kernel void RecalculateNormalFlips(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant RecalculateNormalsPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    if (i >= pc.SelectionCount) return;
    const NormalRecalcContext ctx{b, pc};
    const uint face = ctx.Run(pc.Selection)[i];
    device const uint *component = ctx.Component(ctx.Root(face));
    ctx.Run(pc.Flips)[i] = component[NormalComponentOpen] ? 0u : ctx.Parity(face) ^ uint(Read64(component + NormalComponentVolume) < 0);
}
