#include "ConnectivityRead.metal"
#include "gpu/NeighborhoodPushConstants.h"

// Listed element i's record.
inline NeighborhoodRecord RecordOf(device const BindlessSet &b, constant NeighborhoodPushConstants &pc, uint i) {
    if (pc.Records == InvalidOffset) return {pc.Connectivity.Tables, pc.FacePrimitives};
    device const uint *scratch = BindlessBuffer(uint, b.Buffer, pc.ScratchSlot);
    return reinterpret_cast<device const NeighborhoodRecord *>(scratch + pc.RecordTable)[scratch[pc.Records + i]];
}
inline ConnectivityRef WithTables(ConnectivityRef at, ElementTables tables) {
    at.Tables = tables;
    return at;
}

struct GatherContext {
    device const BindlessSet &B;
    constant NeighborhoodPushConstants &Pc;
    NeighborhoodRecord Record;
    ConnectivityView View;

    // The context of listed element i, or of the listing's one record for a pass over no single element.
    GatherContext(device const BindlessSet &b, constant NeighborhoodPushConstants &pc, uint i = 0u)
        : B(b), Pc(pc), Record(RecordOf(b, pc, i)), View{b, WithTables(pc.Connectivity, Record.Tables)} {}
    device uint *Run(uint offset) const { return BindlessBufferMutable(uint, B.Buffer, Pc.ScratchSlot) + offset; }
    uint Element(uint i) const { return Run(Pc.Elements)[i]; }
};

kernel void NeighborhoodFaceCount(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant NeighborhoodPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    if (i >= pc.ElementCount) return;
    const GatherContext ctx{b, pc, i};
    ctx.Run(pc.Counts)[i] = ctx.View.Valence(ctx.Element(i));
}
kernel void NeighborhoodFaceFill(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant NeighborhoodPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    if (i >= pc.ElementCount) return;
    const GatherContext ctx{b, pc, i};
    const uint face = ctx.Element(i);
    const uint2 range = ctx.View.FaceHalfedges(face);
    device uint *items = ctx.Run(pc.Items) + 9u * ctx.Run(pc.Counts)[i];
    for (uint h = range.x; h < range.y; ++h, items += 9u) {
        const uint opposite = ctx.View.Opposite(h), e = ctx.View.Edge(h), other = ctx.View.OtherSide(h, face);
        items[0] = h;
        items[1] = ctx.View.VertexOf(h);
        items[2] = e;
        items[3] = opposite;
        items[4] = opposite == InvalidOffset ? InvalidOffset : ctx.View.HalfedgeFace(opposite);
        items[5] = ctx.View.EdgeSides(e);
        items[6] = other;
        items[7] = other == InvalidOffset ? InvalidOffset : ctx.View.HalfedgeFace(other);
        items[8] = other == InvalidOffset ? InvalidOffset : ctx.View.VertexOf(other);
    }
}

// Each listed corner records its vertex, origin, edge, face, the corners after and before it, and its opposite.
kernel void NeighborhoodCorners(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant NeighborhoodPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    if (i >= pc.ElementCount) return;
    const GatherContext ctx{b, pc, i};
    const uint h = ctx.Element(i), face = ctx.View.HalfedgeFace(h);
    device uint *items = ctx.Run(pc.Items) + 7u * i;
    items[0] = ctx.View.VertexOf(h);
    items[1] = ctx.View.From(h);
    items[2] = ctx.View.Edge(h);
    items[3] = face;
    items[4] = ctx.View.Next(h, face);
    items[5] = ctx.View.Previous(h);
    items[6] = ctx.View.Opposite(h);
}

// Nonnegative floats order as their bits, so each simdgroup raises the extent word to its largest magnitude.
kernel void NeighborhoodExtent(JOB_TABLE_ARGS(VertexExtentJob)) {
    JOB_THREAD(job, i);
    const ConnectivityView view{b, job.Connectivity};
    const uint v = BlockHandle(i, b, job.Blocks);
    const float3 p = v == InvalidOffset ? float3(0.f) : abs(float3(BindlessBuffer(Vertex, b.VertexBuffer, job.VertexSlot)[view.VertexAddress(v)].Position));
    const float largest = simd_max(max(0.f, max(p.x, max(p.y, p.z))));
    if (simd_is_first()) atomic_fetch_max_explicit(reinterpret_cast<device atomic_uint *>(BindlessBufferMutable(uint, b.Buffer, job.Extent.Slot) + job.Extent.Offset), as_type<uint>(largest), memory_order_relaxed);
}

kernel void NeighborhoodFacePrimitives(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant NeighborhoodPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    if (i >= pc.ElementCount) return;
    const GatherContext ctx{b, pc, i};
    ctx.Run(pc.Items)[i] = BindlessBuffer(uint, b.ElementPrimitiveBuffer, ctx.Record.FacePrimitives.Slot)[ctx.Record.FacePrimitives.Offset + ctx.Element(i)];
}
kernel void NeighborhoodVertexNormals(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant NeighborhoodPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    if (i >= pc.ElementCount) return;
    const GatherContext ctx{b, pc, i};
    device const packed_float3 *normals = BindlessBuffer(packed_float3, b.Buffer, pc.VertexNormalSlot);
    reinterpret_cast<device packed_float3 *>(ctx.Run(pc.Positions))[i] = normals[ctx.View.VertexAddress(ctx.Element(i))];
}
kernel void NeighborhoodFaceNormals(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant NeighborhoodPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    if (i >= pc.ElementCount) return;
    const GatherContext ctx{b, pc, i};
    device const packed_float3 *normals = BindlessBuffer(packed_float3, b.Buffer, pc.FaceNormalSlot);
    reinterpret_cast<device packed_float3 *>(ctx.Run(pc.Positions))[i] = normals[ctx.View.FaceAddress(ctx.Element(i))];
}
// Each listed edge's ends, from its first halfedge's origin to its vertex.
kernel void NeighborhoodEdgeEnds(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant NeighborhoodPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    if (i >= pc.ElementCount) return;
    const GatherContext ctx{b, pc, i};
    const uint2 ends = ctx.View.EdgeEnds(ctx.Element(i));
    device uint *items = ctx.Run(pc.Items) + 2u * i;
    items[0] = ends.x;
    items[1] = ends.y;
}

kernel void NeighborhoodVertexEdgeCount(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant NeighborhoodPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    if (i >= pc.ElementCount) return;
    const GatherContext ctx{b, pc, i};
    uint count = 0u;
    ctx.View.ForEachIncidentEdge(ctx.Element(i), [&](uint) { ++count; });
    ctx.Run(pc.Counts)[i] = count;
}
kernel void NeighborhoodVertexEdgeFill(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant NeighborhoodPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    if (i >= pc.ElementCount) return;
    const GatherContext ctx{b, pc, i};
    const uint v = ctx.Element(i);
    device uint *items = ctx.Run(pc.Items) + 5u * ctx.Run(pc.Counts)[i];
    ctx.View.ForEachIncidentEdge(v, [&](uint e) {
        const uint h = ctx.View.EdgeHalfedge(e);
        items[0] = e;
        items[1] = ctx.View.OtherEnd(h, v);
        items[2] = ctx.View.EdgeSides(e);
        items[3] = h;
        items[4] = ctx.View.HalfedgeFace(h);
        items += 5u;
    });
}

kernel void NeighborhoodVertexCornerCount(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant NeighborhoodPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    if (i >= pc.ElementCount) return;
    const GatherContext ctx{b, pc, i};
    ctx.Run(pc.Counts)[i] = ctx.View.Incoming(ctx.Element(i)).y;
}
kernel void NeighborhoodVertexCornerFill(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant NeighborhoodPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    if (i >= pc.ElementCount) return;
    const GatherContext ctx{b, pc, i};
    device uint *items = ctx.Run(pc.Items) + 3u * ctx.Run(pc.Counts)[i];
    for (const auto item : ctx.View.Fan(ctx.Element(i))) {
        items[0] = item.x;
        items[1] = item.y;
        items[2] = item.y == InvalidOffset ? 0u : ctx.View.Valence(item.y);
        items += 3u;
    }
}

kernel void NeighborhoodPositions(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant NeighborhoodPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    if (i >= pc.ElementCount) return;
    const GatherContext ctx{b, pc, i};
    device const Vertex *vertices = BindlessBuffer(Vertex, b.VertexBuffer, pc.VertexSlot);
    reinterpret_cast<device packed_float3 *>(ctx.Run(pc.Positions))[i] = vertices[ctx.View.VertexAddress(ctx.Element(i))].Position;
}

// The nearest live vertex to each query point among its candidate blocks, the lowest handle on ties, reduced in one threadgroup per point.
kernel void NeighborhoodNearest(uint point [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant NeighborhoodPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    const GatherContext ctx{b, pc};
    device const Vertex *vertices = BindlessBuffer(Vertex, b.VertexBuffer, pc.VertexSlot);
    device const MeshElementBlock *blocks = BindlessBuffer(MeshElementBlock, b.Buffer, pc.VertexBlocksSlot);
    const float3 query = float3(reinterpret_cast<device const packed_float3 *>(ctx.Run(pc.Positions))[point]);
    device const uint *first = ctx.Run(pc.Counts) + point;
    float nearest = INFINITY;
    uint handle = InvalidOffset;
    for (uint at = first[0]; at < first[1]; ++at) {
        const uint block = ctx.Element(at);
        if (!SlotLive(blocks[block], lane)) continue;
        const float3 d = float3(vertices[block * MeshElementBlockSize + lane].Position) - query;
        const float distance = dot(d, d);
        const uint candidate = blocks[block].Local * MeshElementBlockSize + lane;
        if (distance < nearest || (distance == nearest && candidate < handle)) {
            nearest = distance;
            handle = candidate;
        }
    }
    threadgroup float distances[256];
    threadgroup uint handles[256];
    distances[lane] = nearest;
    handles[lane] = handle;
    for (uint stride = 128u; stride; stride /= 2u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < stride) {
            const float other = distances[lane + stride];
            const uint other_handle = handles[lane + stride];
            if (other < distances[lane] || (other == distances[lane] && other_handle < handles[lane])) {
                distances[lane] = other;
                handles[lane] = other_handle;
            }
        }
    }
    if (lane == 0u) {
        device uint *out = ctx.Run(pc.Items) + 2u * point;
        out[0] = handles[0];
        out[1] = as_type<uint>(distances[0]);
    }
}
