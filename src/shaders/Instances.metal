#include "Bindless.metal"
#include "TRSUtils.metal"
#include "gpu/InstanceJob.h"
#include "gpu/MeshElementBlock.h"
#include "gpu/Vertex.h"

kernel void InstanceTransforms(JOB_TABLE_ARGS(InstanceTransformsJob)) {
    JOB_TILE(pc);
    device const uint *list = BindlessBuffer(uint, b.Buffer, pc.Blocks.Slot) + pc.Blocks.Offset;
    const uint block = list[tile];
    const MeshElementBlock members = BindlessBuffer(MeshElementBlock, b.Buffer, pc.MembershipSlot)[block];
    if (!SlotLive(members, lane)) return;
    uint rank = tile ? list[pc.BlockCount + tile - 1u] : 0u;
    for (uint w = 0u; w < lane / 32u; ++w) rank += popcount(members.Live[w]);
    rank += popcount(members.Live[lane / 32u] & ((1u << (lane % 32u)) - 1u));
    const uint handle = members.Local * MeshElementBlockSize + lane;
    const float4 q = BindlessBuffer(packed_float4, b.Buffer, pc.Rotations.ValuesSlot)[pc.Rotations.Offset + handle];
    const float length = precise::sqrt(fma(q.x, q.x, q.y * q.y) + fma(q.z, q.z, q.w * q.w));
    const float4 rotation = length <= 0.f ? float4(0.f, 0.f, 0.f, 1.f) : float4(precise::divide(q.x, length), precise::divide(q.y, length), precise::divide(q.z, length), precise::divide(q.w, length));
    reinterpret_cast<device Transform *>(BindlessBufferMutable(uint, b.Buffer, pc.Transforms.Slot) + pc.Transforms.Offset)[rank] = {
        BindlessBuffer(Vertex, b.VertexBuffer, pc.VertexSlot)[block * MeshElementBlockSize + lane].Position,
        packed_float4(rotation),
        BindlessBuffer(packed_float3, b.Buffer, pc.Scales.ValuesSlot)[pc.Scales.Offset + handle],
    };
}

kernel void PlaceCopies(JOB_TABLE_ARGS(PlaceCopiesJob)) {
    JOB_THREAD(pc, i);
    if (i >= pc.Count) return;
    const uint address = pc.First + i;
    if (!SlotLive(BindlessBuffer(MeshElementBlock, b.Buffer, pc.MembershipSlot)[address / MeshElementBlockSize], address % MeshElementBlockSize)) return;
    const Transform transform = reinterpret_cast<device const Transform *>(BindlessBuffer(uint, b.Buffer, pc.Transforms.Slot) + pc.Transforms.Offset)[i / pc.Bound];
    device Vertex &placed = BindlessBufferMutable(Vertex, b.VertexBuffer, pc.VertexSlot)[address];
    placed.Position = packed_float3(trs_transform_point(transform, float3(placed.Position)));
}
