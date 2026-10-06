#include "ElementWorkShared.metal"
#include "gpu/BindlessBindings.h"
#include "gpu/SpatialFaceQueryPushConstants.h"
#include "gpu/Vertex.h"

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
    if (pc.Mode != 2u) {
        float3 sum = float3(0.f);
        bool negative = false, positive = false;
        for (uint h = first; h < last; ++h) {
            const uint handle = corners[h];
            if (handle >= pc.VertexCapacity) {
                atomic_store_explicit(result + 1u, 1u, memory_order_relaxed);
                return false;
            }
            const float3 position = float3(vertices[handle].Position);
            sum += position;
            const float distance = dot(float3(pc.PlaneNormal), position) - pc.PlaneOffset;
            negative |= distance < 0.f;
            positive |= distance > 0.f;
        }
        // Tangencies and already cut boundaries add no new face interior.
        // A straddling face still qualifies when another vertex lies on the plane.
        return pc.Mode == 1u ? dot(float3(pc.PlaneNormal), sum / float(last - first)) - pc.PlaneOffset < 0.f : negative && positive;
    }
    const float4x4 to_clip = pc.ScreenTransform.Unpack();
    for (uint h = first; h < last; ++h) {
        const uint from = corners[h == first ? last - 1u : h - 1u], to = corners[h];
        if (from >= pc.VertexCapacity || to >= pc.VertexCapacity) {
            atomic_store_explicit(result + 1u, 1u, memory_order_relaxed);
            return false;
        }
        const float3 pa = float3(vertices[from].Position), pb = float3(vertices[to].Position);
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

// Evaluate canonical candidate faces directly, without render ownership.
kernel void SpatialFaceQueryExpand(device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant SpatialFaceQueryPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= pc.CandidateCount) return;
    const uint face = WorkGroupElement(b, pc.Candidates, index);
    device atomic_uint *result = BindlessBufferMutable(atomic_uint, b.Buffer, pc.ResultSlot) + pc.ResultOffset;
    if (face == InvalidOffset) { atomic_store_explicit(result + 1u, 1u, memory_order_relaxed); return; }
    if (SpatialFaceAffected(b, pc, face, result)) MarkWork(b, pc.Faces, face);
}
