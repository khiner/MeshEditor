#include "ConnectivityRead.metal"
#include "gpu/NeighborhoodPushConstants.h"

struct GatherContext {
    device const BindlessSet &B;
    constant NeighborhoodPushConstants &Pc;
    ConnectivityView View;

    GatherContext(device const BindlessSet &b, constant NeighborhoodPushConstants &pc)
        : B(b), Pc(pc), View{b, pc.Connectivity} {}
    device uint *Run(uint offset) const { return BindlessBufferMutable(uint, B.Buffer, Pc.ScratchSlot) + offset; }
    uint Element(uint i) const { return Run(Pc.Elements)[i]; }
};

kernel void NeighborhoodFaceCount(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant NeighborhoodPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    if (i >= pc.ElementCount) return;
    const GatherContext ctx{b, pc};
    ctx.Run(pc.Counts)[i] = ctx.View.Valence(ctx.Element(i));
}
kernel void NeighborhoodFaceFill(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant NeighborhoodPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    if (i >= pc.ElementCount) return;
    const GatherContext ctx{b, pc};
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
    const GatherContext ctx{b, pc};
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

kernel void NeighborhoodExtent(uint lane [[thread_index_in_threadgroup]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant NeighborhoodPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    const GatherContext ctx{b, pc};
    device const Vertex *vertices = BindlessBuffer(Vertex, b.VertexBuffer, pc.VertexSlot);
    threadgroup float partial[256];
    float largest = 0.f;
    for (uint i = lane; i < pc.ElementCount; i += 256u) {
        const float3 p = abs(float3(vertices[ctx.View.VertexAddress(ctx.Element(i))].Position));
        largest = max(largest, max(p.x, max(p.y, p.z)));
    }
    partial[lane] = largest;
    for (uint stride = 128u; stride; stride /= 2u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < stride) partial[lane] = max(partial[lane], partial[lane + stride]);
    }
    if (lane == 0u) ctx.Run(pc.Positions)[0] = as_type<uint>(partial[0]);
}

kernel void NeighborhoodFacePrimitives(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant NeighborhoodPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    if (i >= pc.ElementCount) return;
    const GatherContext ctx{b, pc};
    ctx.Run(pc.Items)[i] = BindlessBuffer(uint, b.ElementPrimitiveBuffer, pc.FacePrimitives.Slot)[pc.FacePrimitives.Offset + ctx.Element(i)];
}
kernel void NeighborhoodVertexNormals(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant NeighborhoodPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    if (i >= pc.ElementCount) return;
    const GatherContext ctx{b, pc};
    device const packed_float3 *normals = BindlessBuffer(packed_float3, b.Buffer, pc.VertexNormalSlot);
    reinterpret_cast<device packed_float3 *>(ctx.Run(pc.Positions))[i] = normals[ctx.View.VertexAddress(ctx.Element(i))];
}
kernel void NeighborhoodFaceNormals(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant NeighborhoodPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    if (i >= pc.ElementCount) return;
    const GatherContext ctx{b, pc};
    device const packed_float3 *normals = BindlessBuffer(packed_float3, b.Buffer, pc.FaceNormalSlot);
    reinterpret_cast<device packed_float3 *>(ctx.Run(pc.Positions))[i] = normals[ctx.View.FaceAddress(ctx.Element(i))];
}

kernel void NeighborhoodVertexEdgeCount(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant NeighborhoodPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    if (i >= pc.ElementCount) return;
    const GatherContext ctx{b, pc};
    uint count = 0u;
    ctx.View.ForEachIncidentEdge(ctx.Element(i), [&](uint) { ++count; });
    ctx.Run(pc.Counts)[i] = count;
}
kernel void NeighborhoodVertexEdgeFill(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant NeighborhoodPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    if (i >= pc.ElementCount) return;
    const GatherContext ctx{b, pc};
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
    const GatherContext ctx{b, pc};
    ctx.Run(pc.Counts)[i] = ctx.View.Incoming(ctx.Element(i)).y;
}
kernel void NeighborhoodVertexCornerFill(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant NeighborhoodPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    if (i >= pc.ElementCount) return;
    const GatherContext ctx{b, pc};
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
    const GatherContext ctx{b, pc};
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
