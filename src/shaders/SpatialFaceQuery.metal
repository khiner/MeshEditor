#include "MeshletIndexShared.metal"
#include "ElementWorkShared.metal"
#include "gpu/BindlessBindings.h"
#include "gpu/SpatialFaceQueryPushConstants.h"
#include "gpu/MeshletSpatialNode.h"
#include "gpu/MeshletRecord.h"
#include "gpu/MeshElementBlock.h"
#include "gpu/Vertex.h"

inline bool StrokeTouchesRect(float2 a, float2 b, float2 lo, float2 hi) {
    const float2 d = b - a;
    float enter = 0.f, leave = 1.f;
    for (uint axis = 0u; axis < 2u; ++axis) {
        if (abs(d[axis]) < 1e-12f) {
            if (a[axis] < lo[axis] || a[axis] > hi[axis]) return false;
        } else {
            float near = (lo[axis] - a[axis]) / d[axis];
            float far = (hi[axis] - a[axis]) / d[axis];
            if (near > far) { const float tmp = near; near = far; far = tmp; }
            enter = max(enter, near); leave = min(leave, far);
            if (enter > leave) return false;
        }
    }
    return true;
}

inline bool SpatialFaceOverlap(const SpatialFaceQueryPushConstants pc, float3 center, float radius) {
    if (pc.Mode != 2u) {
        const float distance = dot(float3(pc.PlaneNormal), center) - pc.PlaneOffset;
        const float margin = 1e-5f * (1.f + radius);
        const float plane_radius = radius * length(float3(pc.PlaneNormal));
        return pc.Mode == 0u ? abs(distance) <= plane_radius + margin : distance < plane_radius + margin;
    }
    const float4x4 transform = pc.ScreenTransform.Unpack();
    const float2 extent = float2(pc.Extent);
    float2 lo = float2(INFINITY), hi = float2(-INFINITY);
    // A projected convex box contains the projection of its inscribed sphere
    // when all corners are in front of the camera. Near-plane overlap is kept.
    for (uint corner = 0u; corner < 8u; ++corner) {
        const float3 delta = float3(corner & 1u ? radius : -radius,
                                    corner & 2u ? radius : -radius,
                                    corner & 4u ? radius : -radius);
        const float4 clip = transform * float4(center + delta, 1.f);
        if (clip.w <= 1e-5f) return true;
        const float2 pixel = float2(clip.x / clip.w + 1.f, 1.f - clip.y / clip.w) * 0.5f * extent;
        lo = min(lo, pixel); hi = max(hi, pixel);
    }
    const float2 margin = float2(1e-3f);
    return StrokeTouchesRect(float2(pc.KnifeStart), float2(pc.KnifeEnd), lo - margin, hi + margin);
}

inline bool SpatialNodeOverlap(const SpatialFaceQueryPushConstants pc, MeshletSpatialNode node) {
    const float3 center=(float3(node.Box.Min)+float3(node.Box.Max))*0.5f;
    const float radius=length((float3(node.Box.Max)-float3(node.Box.Min))*0.5f);
    return SpatialFaceOverlap(pc,center,radius);
}

template<bool Mark>
inline void GatherSpatialMeshlet(device const BindlessSet &b,const SpatialFaceQueryPushConstants pc,uint handle) {
    device atomic_uint *result = BindlessBufferMutable(atomic_uint, b.Buffer, pc.ResultSlot) + pc.ResultOffset;
    if (handle>=pc.MeshletCapacity || MeshletIndexRank(b,pc.Meshlets,handle)==InvalidOffset) {
        atomic_store_explicit(result + 1u,1u,memory_order_relaxed); return;
    }
    const MeshletRecord meshlet=BindlessBuffer(MeshletRecord,b.Buffer,pc.MeshletSlot)[handle];
    if (meshlet.RefinedGroup!=InvalidOffset || meshlet.Topology!=0u) {
        atomic_store_explicit(result + 1u,1u,memory_order_relaxed); return;
    }
    if (!SpatialFaceOverlap(pc,float3(meshlet.Center),meshlet.Radius)) return;
    if constexpr (!Mark) {
        const uint first=atomic_fetch_add_explicit(result,meshlet.TriangleCount,memory_order_relaxed);
        if (ulong(first)+meshlet.TriangleCount>UINT_MAX) atomic_store_explicit(result+1u,1u,memory_order_relaxed);
        atomic_fetch_add_explicit(result+3u,1u,memory_order_relaxed);
    } else {
        const uint base=atomic_fetch_add_explicit(result+2u,meshlet.TriangleCount,memory_order_relaxed);
        const uint index=atomic_fetch_add_explicit(result+3u,1u,memory_order_relaxed);
        if (ulong(base)+meshlet.TriangleCount>pc.CandidateCount || index>=pc.MeshletCandidateCount) {
            atomic_store_explicit(result+1u,1u,memory_order_relaxed); return;
        }
        BindlessBufferMutable(uint,b.Buffer,pc.MeshletCandidates.Slot)[pc.MeshletCandidates.Offset+index]=handle;
    }
}

template<bool Mark>
inline void VisitSpatialFaces(device const BindlessSet &b,const SpatialFaceQueryPushConstants pc,uint seed) {
    device const MeshletSpatialNode *nodes=BindlessBuffer(MeshletSpatialNode,b.Buffer,pc.SpatialNodeSlot);
    uint current=seed,previous=InvalidOffset;
    for (;;) {
        if (current>=pc.SpatialNodeCapacity) return;
        const MeshletSpatialNode node=nodes[current];
        uint next=node.Parent;
        if (previous==node.Parent || previous==InvalidOffset) {
            if (SpatialNodeOverlap(pc,node)) {
                GatherSpatialMeshlet<Mark>(b,pc,node.Meshlet);
                if (node.Left!=InvalidOffset) next=node.Left;
                else if (node.Right!=InvalidOffset) next=node.Right;
            }
        } else if (previous==node.Left) {
            if (node.Right!=InvalidOffset) next=node.Right;
        }
        if (current==seed && next==node.Parent) return;
        previous=current;
        current=next;
    }
}

template<bool Mark>
inline void QuerySpatialFaces(device const BindlessSet &b, const SpatialFaceQueryPushConstants pc,
                              uint group, uint lane) {
    device const MeshletSpatialNode *nodes = BindlessBuffer(MeshletSpatialNode,b.Buffer,pc.SpatialNodeSlot);
    const uint path=group*256u+lane;
    uint current=pc.SpatialRoot;
    for (uint depth=0u;depth<pc.SeedDepth;++depth) {
        if (current>=pc.SpatialNodeCapacity) return;
        const auto node=nodes[current];
        if (!SpatialNodeOverlap(pc,node)) return;
        if (!(path & ((1u<<(pc.SeedDepth-depth))-1u))) GatherSpatialMeshlet<Mark>(b,pc,node.Meshlet);
        const uint bit=1u<<(pc.SeedDepth-depth-1u);
        current=path&bit ? node.Right : node.Left;
        if (current==InvalidOffset) return;
    }
    VisitSpatialFaces<Mark>(b,pc,current);
}

kernel void SpatialFaceQueryCount(device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant SpatialFaceQueryPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    QuerySpatialFaces<false>(b, pc, group, lane);
}

kernel void SpatialFaceQueryGather(device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant SpatialFaceQueryPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    QuerySpatialFaces<true>(b, pc, group, lane);
}

inline bool SpatialFaceAffected(device const BindlessSet &b, const SpatialFaceQueryPushConstants pc,
                                uint face, device atomic_uint *result) {
    if (face >= pc.FaceRangeCapacity) {
        atomic_store_explicit(result + 1u, 1u, memory_order_relaxed);
        return false;
    }
    device const uint *ranges = BindlessBuffer(uint, b.Buffer, pc.FaceRangeSlot);
    device const uint *corners = BindlessBuffer(uint, b.IndexBuffer, pc.CornerSlot);
    device const Vertex *vertices = BindlessBuffer(Vertex, b.VertexBuffer, pc.VertexSlot);
    const uint first = ranges[2u * face], last = ranges[2u * face + 1u];
    if (first >= last || last > pc.CornerCapacity) {
        atomic_store_explicit(result + 1u, 1u, memory_order_relaxed);
        return false;
    }
    if (pc.Mode == 1u) {
        float3 sum = float3(0.f);
        for (uint h = first; h < last; ++h) {
            const uint handle = corners[h];
            if (handle >= pc.VertexCapacity) {
                atomic_store_explicit(result + 1u, 1u, memory_order_relaxed);
                return false;
            }
            sum += float3(vertices[handle].Position);
        }
        return dot(float3(pc.PlaneNormal), sum / float(last - first)) - pc.PlaneOffset < 0.f;
    }
    const float4x4 to_clip = pc.Mode == 2u ? pc.ScreenTransform.Unpack() : float4x4(1.f);
    for (uint h = first; h < last; ++h) {
        const uint from = corners[h == first ? last - 1u : h - 1u], to = corners[h];
        if (from >= pc.VertexCapacity || to >= pc.VertexCapacity) {
            atomic_store_explicit(result + 1u, 1u, memory_order_relaxed);
            return false;
        }
        const float3 pa = float3(vertices[from].Position), pb = float3(vertices[to].Position);
        if (pc.Mode == 0u) {
            const float a = dot(float3(pc.PlaneNormal), pa) - pc.PlaneOffset;
            const float c = dot(float3(pc.PlaneNormal), pb) - pc.PlaneOffset;
            if ((a < 0.f) != (c < 0.f) && a != c) return true;
            continue;
        }
        const float4 ca = to_clip * float4(pa, 1.f), cb = to_clip * float4(pb, 1.f);
        if (ca.w <= 0.f || cb.w <= 0.f) continue;
        const float2 extent = float2(pc.Extent);
        const float2 a = float2(ca.x / ca.w + 1.f, 1.f - ca.y / ca.w) * 0.5f * extent;
        const float2 c = float2(cb.x / cb.w + 1.f, 1.f - cb.y / cb.w) * 0.5f * extent;
        const float2 d = c - a, k = float2(pc.KnifeEnd) - float2(pc.KnifeStart);
        const float2 w = float2(pc.KnifeStart) - a;
        const float denominator = d.x * k.y - d.y * k.x;
        if (abs(denominator) < 1e-12f) continue;
        const float t = (w.x * k.y - w.y * k.x) / denominator;
        const float u = (w.x * d.y - w.y * d.x) / denominator;
        if (t > 0.f && t < 1.f && u >= 0.f && u <= 1.f) return true;
    }
    return false;
}

// One group per intersecting finest meshlet. All candidates share the sparse
// work writer used by other local edits, including duplicate face references.
kernel void SpatialFaceQueryExpand(device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant SpatialFaceQueryPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    if (group >= pc.MeshletCandidateCount) return;
    const uint id = BindlessBuffer(uint, b.Buffer, pc.MeshletCandidates.Slot)[pc.MeshletCandidates.Offset + group];
    device atomic_uint *result = BindlessBufferMutable(atomic_uint, b.Buffer, pc.ResultSlot) + pc.ResultOffset;
    if (id>=pc.MeshletCapacity) { atomic_store_explicit(result+1u,1u,memory_order_relaxed); return; }
    const MeshletRecord meshlet=BindlessBuffer(MeshletRecord,b.Buffer,pc.MeshletSlot)[id];
    if (lane>=meshlet.TriangleCount) return;
    if (ulong(meshlet.TriangleOffset)+meshlet.TriangleCount>pc.TriangleIdCapacity) {
        atomic_store_explicit(result+1u,1u,memory_order_relaxed); return;
    }
    const uint triangle=BindlessBuffer(uint,b.Buffer,pc.TriangleIdSlot)[meshlet.TriangleOffset+lane];
    if (triangle>=pc.TriangleCapacity) { atomic_store_explicit(result+1u,1u,memory_order_relaxed); return; }
    const uint corner=BindlessBuffer(packed_uint3,b.Buffer,pc.TriangleSlot)[triangle].x;
    if (corner>=pc.CornerCapacity) { atomic_store_explicit(result+1u,1u,memory_order_relaxed); return; }
    const uint face=BindlessBuffer(uint,b.Buffer,pc.HalfedgeFaceSlot)[corner];
    if (face>=pc.FaceCapacity) { atomic_store_explicit(result+1u,1u,memory_order_relaxed); return; }
    const MeshElementBlock block=BindlessBuffer(MeshElementBlock,b.Buffer,pc.FaceBlockSlot)[face/256u];
    if (block.Owner!=pc.FaceOwner || !(block.Live[(face&255u)/32u]&(1u<<(face&31u)))) {
        atomic_store_explicit(result+1u,1u,memory_order_relaxed); return;
    }
    if (SpatialFaceAffected(b,pc,face,result)) MarkWork(b,pc.Faces,face);
}
