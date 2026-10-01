#include "ConnectivityRead.metal"
#include "ElementWorkShared.metal"
#include "gpu/MeshClosurePushConstants.h"
#include "gpu/MeshElementBlock.h"

inline uint ClosureInput(device const BindlessSet &b, constant MeshClosurePushConstants &pc, uint group, uint lane) {
    const uint i = group * 256u + lane;
    return i < pc.InputBound ? WorkGroupElement(b, pc.Input, i) : InvalidOffset;
}

kernel void MeshClosureCount(
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant MeshClosurePushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]], uint sl [[thread_index_in_simdgroup]]
) {
    const ConnectivityView conn{b, pc.Connectivity, pc.FaceCount};
    const uint element = ClosureInput(b, pc, group, lane);
    uint count = 0u;
    if (element != InvalidOffset && pc.InputDomain == 2u) {
        const uint2 loop = conn.FaceHalfedges(element);
        count = loop.y - loop.x;
    } else if (element != InvalidOffset) {
        const uint h = conn.EdgeHalfedge(element);
        if (h != InvalidOffset) {
            device const uint *corners = BindlessBuffer(uint, b.IndexBuffer, pc.CornerSlot);
            count = conn.Incoming(corners[h]).y + conn.Incoming(corners[conn.Previous(h)]).y;
        }
    }
    const uint total = simd_sum(count);
    if (sl == 0u && total) atomic_fetch_add_explicit(BindlessBufferMutable(atomic_uint, b.Buffer, pc.Incidence.Slot) + pc.Incidence.Offset, total, memory_order_relaxed);
}

kernel void MeshClosureExpand(
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant MeshClosurePushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]
) {
    const ConnectivityView conn{b, pc.Connectivity, pc.FaceCount};
    const uint element = ClosureInput(b, pc, group, lane);
    if (element != InvalidOffset && pc.InputDomain == 0u) {
        for (const auto item : conn.Fan(element)) {
            const uint h = item.x, face = conn.HalfedgeFace(h);
            MarkWork(b, pc.Work[1], h);
            MarkWork(b, pc.Work[2], face);
            MarkWork(b, pc.Work[3], conn.Edge(h));
            // A line corner's pair takes the place of the next corner, so a line's two corners stay together.
            const uint next = face == InvalidOffset ? conn.Previous(h) : conn.Next(h);
            if (next != InvalidOffset) {
                MarkWork(b, pc.Work[1], next);
                MarkWork(b, pc.Work[3], conn.Edge(next));
            }
        }
    } else if (element != InvalidOffset && pc.InputDomain == 3u) {
        // A line holds its two corners and their vertices.
        const uint h = conn.EdgeHalfedge(element), pair = conn.Previous(h);
        device const uint *corners = BindlessBuffer(uint, b.IndexBuffer, pc.CornerSlot);
        MarkWork(b, pc.Work[1], h);
        MarkWork(b, pc.Work[1], pair);
        MarkWork(b, pc.Work[0], corners[h]);
        MarkWork(b, pc.Work[0], corners[pair]);
    } else if (element != InvalidOffset) {
        const uint2 loop = conn.FaceHalfedges(element);
        device const uint *corners = BindlessBuffer(uint, b.IndexBuffer, pc.CornerSlot);
        MarkWork(b, pc.Work[2], element);
        for (uint h = loop.x; h < loop.y; ++h) {
            MarkWork(b, pc.Work[0], corners[h]);
            MarkWork(b, pc.Work[1], h);
            MarkWork(b, pc.Work[3], conn.Edge(h));
        }
    }
    const uint i = group * 256u + lane;
    if (i < pc.RetainedBound) {
        const uint v = WorkGroupElement(b, pc.Retained, i);
        if (v != InvalidOffset) MarkWork(b, pc.Work[0], v);
    }
}

kernel void MeshClosureEdgeVertices(
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant MeshClosurePushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]
) {
    const uint edge = ClosureInput(b, pc, group, lane);
    if (edge == InvalidOffset) return;
    const ConnectivityView conn{b, pc.Connectivity, pc.FaceCount};
    const uint h = conn.EdgeHalfedge(edge);
    if (h == InvalidOffset) return;
    device const uint *corners = BindlessBuffer(uint, b.IndexBuffer, pc.CornerSlot);
    MarkWork(b, pc.Work[0], corners[h]);
    MarkWork(b, pc.Work[0], corners[conn.Previous(h)]);
}

inline void FaceTriangleError(device const BindlessSet &b, constant FaceTrianglePushConstants &pc) {
    atomic_store_explicit(BindlessBufferMutable(atomic_uint, b.Buffer, pc.Error.Slot) + pc.Error.Offset, 1u, memory_order_relaxed);
}
inline bool FaceTriangleOwned(device const BindlessSet &b, uint slot, uint owner, uint handle, uint capacity) {
    if (handle >= capacity) return false;
    const auto block = BindlessBuffer(MeshElementBlock, b.Buffer, slot)[handle / MeshElementBlockSize];
    return block.Owner == owner && (block.Live[(handle % MeshElementBlockSize) / 32u] & (1u << (handle % 32u)));
}

// One simd group per face marks its triangles and adds their count to Total.
kernel void MeshClosureTriangles(
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant FaceTrianglePushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]
) {
    if (group >= pc.FaceBound) return;
    const uint face = WorkGroupElement(b, pc.Faces, group);
    if (face == InvalidOffset) return;
    if (!FaceTriangleOwned(b, pc.FaceBlocksSlot, pc.FaceOwner, face, pc.FaceCapacity)) { FaceTriangleError(b, pc); return; }
    const uint2 corners = uint2(BindlessBuffer(packed_uint2, b.Buffer, pc.FaceRangesSlot)[face]);
    const uint first = BindlessBuffer(uint, b.ObjectIdBuffer, pc.FaceTrianglesSlot)[face];
    if (corners.y < corners.x || corners.y - corners.x < 3u || corners.y > pc.CornerCapacity ||
        ulong(first) + corners.y - corners.x - 2u > pc.TriangleCapacity) {
        FaceTriangleError(b, pc);
        return;
    }
    const uint count = corners.y - corners.x - 2u;
    if (lane == 0u) atomic_fetch_add_explicit(BindlessBufferMutable(atomic_uint, b.Buffer, pc.Total.Slot) + pc.Total.Offset, count, memory_order_relaxed);
    for (uint i = lane; i < count; i += 32u) {
        const uint t = first + i;
        if (!FaceTriangleOwned(b, pc.TriangleBlocksSlot, pc.TriangleOwner, t, pc.TriangleCapacity)) {
            FaceTriangleError(b, pc);
            continue;
        }
        const uint3 triangle = uint3(BindlessBuffer(packed_uint3, b.Buffer, pc.TrianglesSlot)[t]);
        if (any(triangle < corners.x) || any(triangle >= corners.y)) { FaceTriangleError(b, pc); continue; }
        MarkWork(b, pc.Triangles, t);
    }
}
