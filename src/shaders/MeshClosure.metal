#include "ConnectivityRead.metal"
#include "ElementWorkShared.metal"
#include "gpu/MeshClosureJob.h"

inline uint ClosureInput(device const BindlessSet &b, constant MeshClosureJob &job, uint i) {
    return i < job.InputBound ? WorkGroupElement(b, job.Input, i) : InvalidOffset;
}

kernel void MeshClosureCount(JOB_TABLE_ARGS(MeshClosureJob), uint sl [[thread_index_in_simdgroup]]) {
    JOB_THREAD(job, i);
    const ConnectivityView conn{b, job.Connectivity};
    const uint element = ClosureInput(b, job, i);
    uint count = 0u;
    if (element != InvalidOffset && job.InputDomain == 2u) {
        const uint2 loop = conn.FaceHalfedges(element);
        count = loop.y - loop.x;
    } else if (element != InvalidOffset) {
        const uint h = conn.EdgeHalfedge(element);
        if (h != InvalidOffset) count = conn.Incoming(conn.VertexOf(h)).y + conn.Incoming(conn.From(h)).y;
    }
    const uint total = simd_sum(count);
    if (sl == 0u && total) atomic_fetch_add_explicit(BindlessBufferMutable(atomic_uint, b.Buffer, job.Incidence.Slot) + job.Incidence.Offset, total, memory_order_relaxed);
}

kernel void MeshClosureExpand(JOB_TABLE_ARGS(MeshClosureJob)) {
    JOB_THREAD(job, i);
    const ConnectivityView conn{b, job.Connectivity};
    const uint element = ClosureInput(b, job, i);
    if (element != InvalidOffset && job.InputDomain == 0u) {
        for (const auto item : conn.Fan(element)) {
            const uint h = item.x, face = conn.HalfedgeFace(h);
            MarkWork(b, job.Work[1], h);
            MarkWork(b, job.Work[2], face);
            MarkWork(b, job.Work[3], conn.Edge(h));
            // A line corner's pair takes the place of the next corner, so a line's two corners stay together.
            const uint next = face == InvalidOffset ? conn.Previous(h) : conn.Next(h);
            if (next != InvalidOffset) {
                MarkWork(b, job.Work[1], next);
                MarkWork(b, job.Work[3], conn.Edge(next));
            }
        }
    } else if (element != InvalidOffset && job.InputDomain == 3u) {
        const uint h = conn.EdgeHalfedge(element);
        if (conn.HalfedgeFace(h) != InvalidOffset) return;
        const uint pair = conn.Previous(h);
        MarkWork(b, job.Work[1], h);
        MarkWork(b, job.Work[1], pair);
        MarkWork(b, job.Work[0], conn.VertexOf(h));
        MarkWork(b, job.Work[0], conn.VertexOf(pair));
        MarkWork(b, job.Work[3], element);
    } else if (element != InvalidOffset) {
        const uint2 loop = conn.FaceHalfedges(element);
        MarkWork(b, job.Work[2], element);
        for (uint h = loop.x; h < loop.y; ++h) {
            MarkWork(b, job.Work[0], conn.VertexOf(h));
            MarkWork(b, job.Work[1], h);
            MarkWork(b, job.Work[3], conn.Edge(h));
        }
    }
    if (i < job.RetainedBound) {
        const uint v = WorkGroupElement(b, job.Retained, i);
        if (v != InvalidOffset && (!job.RetainIsolatedOnly || !conn.Incoming(v).y)) MarkWork(b, job.Work[0], v);
    }
}

kernel void MeshClosureEdgeVertices(JOB_TABLE_ARGS(MeshClosureJob)) {
    JOB_THREAD(job, i);
    const uint edge = ClosureInput(b, job, i);
    if (edge == InvalidOffset) return;
    const ConnectivityView conn{b, job.Connectivity};
    const uint h = conn.EdgeHalfedge(edge);
    if (h == InvalidOffset) return;
    MarkWork(b, job.Work[0], conn.VertexOf(h));
    MarkWork(b, job.Work[0], conn.From(h));
}

inline void FaceTriangleError(device const BindlessSet &b, constant FaceTriangleJob &job) {
    atomic_store_explicit(BindlessBufferMutable(atomic_uint, b.Buffer, job.Error.Slot) + job.Error.Offset, 1u, memory_order_relaxed);
}
// Whether a handle below `bound` is a live element of the set `owner` through its table run.
inline bool FaceTriangleOwned(device const BindlessSet &b, SlotOffset table, uint slot, uint owner, uint handle, uint bound) {
    if (handle >= bound) return false;
    const uint entry = BindlessBuffer(uint, b.Buffer, table.Slot)[table.Offset + handle / MeshElementBlockSize];
    if (entry == InvalidOffset) return false;
    const auto block = BindlessBuffer(MeshElementBlock, b.Buffer, slot)[entry];
    return block.Owner == owner && SlotLive(block, handle % MeshElementBlockSize);
}

kernel void MeshClosureTriangles(JOB_TABLE_ARGS(FaceTriangleJob)) {
    JOB_TILE(job);
    const uint face = WorkGroupElement(b, job.Faces, tile);
    if (face == InvalidOffset) return;
    if (!FaceTriangleOwned(b, job.Tables.Faces, job.FaceBlocksSlot, job.FaceOwner, face, job.FaceHandles)) { FaceTriangleError(b, job); return; }
    const uint address = HandleAddress<MeshElementBlockSize>(b, job.Tables.Faces, face);
    const uint2 corners = uint2(BindlessBuffer(packed_uint2, b.Buffer, job.FaceRangesSlot)[address]);
    const uint first = BindlessBuffer(uint, b.ObjectIdBuffer, job.FaceTrianglesSlot)[address];
    if (corners.y < corners.x || corners.y - corners.x < 3u || corners.y > job.CornerHandles ||
        ulong(first) + corners.y - corners.x - 2u > job.TriangleHandles) {
        FaceTriangleError(b, job);
        return;
    }
    const uint count = corners.y - corners.x - 2u;
    if (lane == 0u) atomic_fetch_add_explicit(BindlessBufferMutable(atomic_uint, b.Buffer, job.Total.Slot) + job.Total.Offset, count, memory_order_relaxed);
    for (uint i = lane; i < count; i += 32u) {
        const uint t = first + i;
        if (!FaceTriangleOwned(b, job.Tables.Triangles, job.TriangleBlocksSlot, job.TriangleOwner, t, job.TriangleHandles)) {
            FaceTriangleError(b, job);
            continue;
        }
        const uint3 triangle = uint3(BindlessBuffer(packed_uint3, b.Buffer, job.TrianglesSlot)[HandleAddress<MeshElementBlockSize>(b, job.Tables.Triangles, t)]);
        if (any(triangle < corners.x) || any(triangle >= corners.y)) { FaceTriangleError(b, job); continue; }
        MarkWork(b, job.Triangles, t);
    }
}
