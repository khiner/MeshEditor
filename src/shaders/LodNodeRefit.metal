#include "Bindless.metal"
#include "EnclosingSphere.metal"
#include "gpu/ClusterGroup.h"
#include "gpu/LodNode.h"
#include "gpu/LodNodeRefitPushConstants.h"

inline void RefitError(device const BindlessSet &b,constant LodNodeRefitPushConstants &pc) {
    atomic_store_explicit(BindlessBufferMutable(atomic_uint,b.Buffer,pc.ErrorSlot),1u,memory_order_relaxed);
}

struct RefitSet {
    device const uint *Table;
    device const MeshElementBlock *Blocks;
    uint Set, BlockCount;
    // The arena address of a live member, or InvalidOffset when the handle names none.
    uint Address(uint handle) const {
        if (handle==InvalidOffset || handle/RenderBlockSize>=BlockCount) return InvalidOffset;
        const uint block=Table[handle/RenderBlockSize], slot=handle%RenderBlockSize;
        return block!=InvalidOffset && Blocks[block].Owner==Set && (Blocks[block].Live[0]>>slot&1u) ? block*RenderBlockSize+slot : InvalidOffset;
    }
};

inline RefitSet ReadRefitSet(device const BindlessSet &b, SetArenaSlots arena, uint set) {
    const MeshElementSet header=set==InvalidOffset ? MeshElementSet{} : BindlessBuffer(MeshElementSet,b.Buffer,arena.Sets)[set];
    return {BindlessBuffer(uint,b.Buffer,arena.Table)+header.First,BindlessBuffer(MeshElementBlock,b.Buffer,arena.Blocks),set,header.BlockCount};
}

// Reads one node's refit samples, with the record's node, member, meshlet and group sets read once per thread.
struct LodRefitReader {
    device const BindlessSet &B;
    constant LodNodeRefitPushConstants &Pc;
    uint Id;
    LodNode Node;
    RefitSet Nodes, Members, Meshlets, Groups;

    EnclosingSphereSample Read(uint i) const {
        if (Node.ChildCount) {
            const uint address=Nodes.Address(Node.ChildOffset+i);
            if (address==InvalidOffset || BindlessBuffer(uint,B.Buffer,Pc.ParentSlot)[address]!=Id) {
                RefitError(B,Pc); return {float4(0.f),0.f};
            }
            const auto value=BindlessBuffer(LodNode,B.Buffer,Pc.NodeSlot)[address];
            if (!value.MeshletCount) return {float4(0.f,0.f,0.f,-1.f),0.f};
            return {float4(float3(value.Center),value.Radius),value.Error};
        }
        const uint member=Members.Address(Node.Members+i/2u);
        const uint cluster=member==InvalidOffset ? InvalidOffset : Meshlets.Address(BindlessBuffer(uint,B.Buffer,Pc.MemberSlot)[member]);
        if (cluster==InvalidOffset) {
            RefitError(B,Pc); return {float4(0.f),0.f};
        }
        const auto record=BindlessBuffer(MeshletRecord,B.Buffer,Pc.MeshletSlot)[cluster];
        if (record.GroupIndex==InvalidOffset) return {float4(float3(record.Center),record.Radius),INFINITY};
        const uint group_address=Groups.Address(record.GroupIndex);
        if (group_address==InvalidOffset) {
            RefitError(B,Pc); return {float4(0.f),0.f};
        }
        const auto group=BindlessBuffer(ClusterGroup,B.Buffer,Pc.GroupSlot)[group_address];
        return {i&1u ? float4(float3(group.Center),group.Radius) : float4(float3(record.Center),record.Radius),group.Error};
    }
};

// The same seven-axis extrema and ordered Ritter expansion as meshoptimizer's computeSphereBounds, then ClusterLod's conservative containment pass.
// The host dispatches only populated nodes that are not pinned finest nodes, and writes their member counts.
kernel void LodNodeRefit(device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant LodNodeRefitPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    uint job_id [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    if (job_id>=pc.Count) return;
    const LodNodeRefitJob job=reinterpret_cast<device const LodNodeRefitJob *>(BindlessBuffer(uint,b.Buffer,pc.Jobs.Slot)+pc.Jobs.Offset)[job_id];
    const uint id=job.Node;
    const RefitSet nodes=ReadRefitSet(b,pc.Nodes,job.NodeSet);
    const uint address=nodes.Address(id);
    if (address==InvalidOffset) {
        if (!tid) RefitError(b,pc);
        return;
    }
    device LodNode &output=BindlessBufferMutable(LodNode,b.Buffer,pc.NodeSlot)[address];
    const LodNode node=output;
    threadgroup_barrier(mem_flags::mem_device);
    if ((node.ChildCount && node.Members!=InvalidOffset) || (!node.ChildCount && node.Members==InvalidOffset)) {
        if (!tid) RefitError(b,pc);
        return;
    }
    const uint members=node.ChildCount ? 0u : node.MeshletCount;
    const uint count=node.ChildCount ? node.ChildCount : members*2u;
    if (!count || members>UINT_MAX/2u) { if (!tid) RefitError(b,pc); return; }
    threadgroup EnclosingSphereScratch scratch;
    const LodRefitReader reader{b,pc,id,node,nodes,ReadRefitSet(b,pc.Members,job.MemberSet),ReadRefitSet(b,pc.Meshlets,job.MeshletSet),ReadRefitSet(b,pc.Groups,job.GroupSet)};
    const auto result=EncloseSpheres(reader,count,tid,lane,simd,scratch);
    if (!tid) {
        output.Center=result.Sphere.xyz;
        output.Radius=result.Sphere.w;
        output.Error=result.Error;
    }
}
