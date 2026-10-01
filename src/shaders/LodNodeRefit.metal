#include "MeshletIndexShared.metal"
#include "EnclosingSphere.metal"
#include "gpu/BindlessBindings.h"
#include "gpu/ClusterGroup.h"
#include "gpu/LodNode.h"
#include "gpu/LodNodeRefitPushConstants.h"
#include "gpu/MeshletRecord.h"

inline void RefitError(device const BindlessSet &b,constant LodNodeRefitPushConstants &pc) {
    atomic_store_explicit(BindlessBufferMutable(atomic_uint,b.Buffer,pc.ErrorSlot),1u,memory_order_relaxed);
}
inline EnclosingSphereSample ReadRefitSample(device const BindlessSet &b,constant LodNodeRefitPushConstants &pc,
                                   uint id, LodNode node, uint i) {
    if (node.ChildCount) {
        const uint child=node.ChildOffset+i;
        if (MeshletIndexRank(b,pc.Nodes,child)==InvalidOffset ||
            BindlessBuffer(uint,b.Buffer,pc.ParentSlot)[child]!=id) {
            RefitError(b,pc); return {float4(0.f),0.f};
        }
        const auto value=BindlessBuffer(LodNode,b.Buffer,pc.NodeSlot)[child];
        if (!value.MeshletCount) return {float4(0.f,0.f,0.f,-1.f),0.f};
        return {float4(float3(value.Center),value.Radius),value.Error};
    }
    const uint cluster=MeshletIndexSelect(b,{pc.Meshlets.NodesSlot,pc.Meshlets.LeavesSlot,node.MeshletRoot},i/2u);
    if (cluster>=pc.MeshletCapacity || MeshletIndexRank(b,pc.Meshlets,cluster)==InvalidOffset) {
        RefitError(b,pc); return {float4(0.f),0.f};
    }
    const auto record=BindlessBuffer(MeshletRecord,b.Buffer,pc.MeshletSlot)[cluster];
    if (record.GroupIndex==InvalidOffset) return {float4(float3(record.Center),record.Radius),INFINITY};
    if (record.GroupIndex>=pc.GroupCapacity || MeshletIndexRank(b,pc.Groups,record.GroupIndex)==InvalidOffset) {
        RefitError(b,pc); return {float4(0.f),0.f};
    }
    const auto group=BindlessBuffer(ClusterGroup,b.Buffer,pc.GroupSlot)[record.GroupIndex];
    return {i&1u ? float4(float3(group.Center),group.Radius) : float4(float3(record.Center),record.Radius),group.Error};
}

struct LodRefitReader {
    device const BindlessSet &B;
    constant LodNodeRefitPushConstants &Pc;
    uint Id;
    LodNode Node;
    EnclosingSphereSample Read(uint i) const { return ReadRefitSample(B,Pc,Id,Node,i); }
};

// The same seven-axis extrema and ordered Ritter expansion as meshoptimizer's computeSphereBounds, then ClusterLod's conservative containment pass.
// The host dispatches only populated nodes that are not pinned finest nodes, and writes their member counts.
kernel void LodNodeRefit(device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant LodNodeRefitPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    uint job_id [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    if (job_id>=pc.Count) return;
    const uint id=BindlessBuffer(uint,b.Buffer,pc.Jobs.Slot)[pc.Jobs.Offset+job_id];
    if (id>=pc.NodeCapacity || MeshletIndexRank(b,pc.Nodes,id)==InvalidOffset) {
        if (!tid) RefitError(b,pc);
        return;
    }
    device LodNode &output=BindlessBufferMutable(LodNode,b.Buffer,pc.NodeSlot)[id];
    const LodNode node=output;
    threadgroup_barrier(mem_flags::mem_device);
    if (ulong(node.ChildOffset)+node.ChildCount>pc.NodeCapacity || (node.ChildCount && node.MeshletRoot!=InvalidOffset) ||
        (!node.ChildCount && (node.MeshletRoot==InvalidOffset || node.MeshletRoot>=pc.IndexNodeCapacity))) {
        if (!tid) RefitError(b,pc);
        return;
    }
    const uint members=node.ChildCount ? 0u : BindlessBuffer(MeshletIndexNode,b.Buffer,pc.Meshlets.NodesSlot)[node.MeshletRoot].Count;
    const uint count=node.ChildCount ? node.ChildCount : members*2u;
    if (!count || members>UINT_MAX/2u) { if (!tid) RefitError(b,pc); return; }
    threadgroup EnclosingSphereScratch scratch;
    const LodRefitReader reader{b,pc,id,node};
    const auto result=EncloseSpheres(reader,count,tid,lane,simd,scratch);
    if (!tid) {
        output.Center=result.Sphere.xyz;
        output.Radius=result.Sphere.w;
        output.Error=result.Error;
    }
}
