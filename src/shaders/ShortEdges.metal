#include "ConnectivityRead.metal"
#include "gpu/ShortEdgesJob.h"

struct ShortEdgeContext {
    device const BindlessSet &B;
    ShortEdgesJob Job;
    ConnectivityView View;
    device const Vertex *Vertices;
    ShortEdgeContext(device const BindlessSet &b, ShortEdgesJob job) : B(b), Job(job), View{b, job.Connectivity}, Vertices(BindlessBuffer(Vertex, b.VertexBuffer, job.VertexSlot)) {}
    // The live halfedge at thread i when it is its edge's first, or InvalidOffset.
    uint EdgeHalfedgeAt(uint i) const {
        const uint h = BlockHandle(i, B, Job.Blocks);
        return h == InvalidOffset || View.Edge(h) == InvalidOffset || !View.EdgeFirst(h) ? InvalidOffset : h;
    }
    float3 Position(uint v) const { return float3(Vertices[View.VertexAddress(v)].Position); }
};

kernel void FlagShortEdges(JOB_TABLE_ARGS(ShortEdgesJob)) {
    JOB_THREAD(job, i);
    const ShortEdgeContext ctx{b, job};
    const uint h = ctx.EdgeHalfedgeAt(i);
    if (h == InvalidOffset) return;
    if (distance(ctx.Position(ctx.View.VertexOf(h)), ctx.Position(ctx.View.From(h))) >= ctx.Job.Distance) return;
    BindlessBufferMutable(uint, b.Buffer, ctx.Job.Flag.Slot)[ctx.Job.Flag.Offset] = 1u;
}

// A vertex next to an end of an edge in a face splits the faces on the edge when it lies within the distance of the edge's interior.
// It must also lie nearer each end than the ends lie to each other.
kernel void FindJunctions(JOB_TABLE_ARGS(ShortEdgesJob)) {
    JOB_THREAD(job, i);
    const ShortEdgeContext ctx{b, job};
    const uint h = ctx.EdgeHalfedgeAt(i);
    if (h == InvalidOffset) return;
    const uint a = ctx.View.From(h), c = ctx.View.VertexOf(h);
    const float3 pa = ctx.Position(a), pb = ctx.Position(c), d = pb - pa;
    const float length = metal::length(d);
    if (!(length > 0.f)) return;
    uint junction = InvalidOffset;
    for (uint side = 0u; side < 2u; ++side) {
        for (const auto item : ctx.View.Fan(side ? c : a)) {
            if (item.y == InvalidOffset) continue;
            const uint neighbors[2] = {ctx.View.VertexOf(ctx.View.Next(item.x, item.y)), ctx.View.VertexOf(ctx.View.Previous(item.x))};
            for (uint k = 0u; k < 2u; ++k) {
                const uint m = neighbors[k];
                if (m == a || m == c || m >= junction) continue;
                const float3 pm = ctx.Position(m);
                const float t = dot(pm - pa, d);
                if (!(t > 0.f && t < length * length) || metal::length(cross(d, pm - pa)) > ctx.Job.Distance * length) continue;
                if (metal::length(pm - pa) < length && metal::length(pm - pb) < length) junction = m;
            }
        }
    }
    if (junction == InvalidOffset) return;
    const uint2 faces = ctx.View.EdgeFaces(ctx.View.Edge(h));
    if (faces.x == InvalidOffset && faces.y == InvalidOffset) return;
    if (faces.x != InvalidOffset && faces.y != InvalidOffset && !ctx.View.FaceHas(faces.x, junction) && !ctx.View.FaceHas(faces.y, junction)) return;
    device uint *list = BindlessBufferMutable(uint, b.Buffer, ctx.Job.Junctions.Slot) + ctx.Job.Junctions.Offset;
    const uint slot = atomic_fetch_add_explicit(AtomicWord(list), 1u, memory_order_relaxed);
    if (slot >= ctx.Job.Capacity) return;
    list[1u + 2u * slot] = ctx.View.Edge(h);
    list[2u + 2u * slot] = junction;
}
