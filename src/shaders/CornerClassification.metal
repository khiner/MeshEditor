#include "NormalSectorTraversal.metal"
#include "ElementWorkShared.metal"
#include "gpu/CornerClassificationPushConstants.h"
#include "gpu/MeshElementBlock.h"

struct CornerClassificationContext {
    device const BindlessSet &B;
    constant CornerClassificationPushConstants &Pc;
    ConnectivityView Conn;
    device atomic_uint *State() const { return BindlessBufferMutable(atomic_uint,B.Buffer,Pc.State.Slot)+Pc.State.Offset; }
    NormalSectorTraversal Sectors() const {
        return {Conn, BindlessBuffer(uchar,B.Buffer,Pc.FaceSharpnessSlot), BindlessBuffer(uchar,B.Buffer,Pc.EdgeSharpnessSlot)};
    }
    bool Flat(uint h) const { return Sectors().Flat(h); }
    bool Sharp(uint h) const { return Sectors().Sharp(h); }
    uint Payload(uint h) const {
        const uint block = BindlessBuffer(uint,B.Buffer,Pc.CornerSectors.BlocksSlot)[h/MeshElementBlockSize];
        return block ? (block-1u)*MeshElementBlockSize+h%MeshElementBlockSize : InvalidOffset;
    }
    uint Root(uint h) const {
        const uint at = Payload(h);
        return at == InvalidOffset ? InvalidOffset : BindlessBuffer(uint,B.Buffer,Pc.CornerSectors.ValuesSlot)[at];
    }
    void Write(uint h, uint root) const {
        const uint at = Payload(h);
        if (at != InvalidOffset) BindlessBufferMutable(uint,B.Buffer,Pc.CornerSectors.ValuesSlot)[at] = root;
    }
    uint Flags(uint2 fan) const {
        uint flags = 0u;
        for (uint i = 0u; i < fan.y; ++i) {
            const uint h = Conn.FanCorner(fan.x+i);
            flags |= Flat(h) ? 1u : 2u;
            if (Sharp(h) || Sharp(Conn.Next(h))) flags |= 4u;
        }
        return flags;
    }
    void Sector(uint root, uint limit) const {
        Write(root,root);
        Sectors().VisitOthers(root,limit,[&](uint h) { Write(h,root); });
    }
};

kernel void CornerClassificationCount(
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant CornerClassificationPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    uint i [[thread_position_in_grid]]
) {
    const uint v = i < pc.VertexCount ? WorkGroupElement(b,pc.Vertices,i) : InvalidOffset;
    if (v == InvalidOffset) return;
    const CornerClassificationContext ctx{b,pc,{b,pc.Connectivity,pc.FaceCount}};
    const auto fan = ctx.Conn.Incoming(v);
    atomic_fetch_add_explicit(ctx.State(),fan.y,memory_order_relaxed);
    atomic_fetch_or_explicit(ctx.State()+1u,ctx.Flags(fan),memory_order_relaxed);
}

kernel void CornerClassificationPlan(
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant CornerClassificationPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    uint i [[thread_position_in_grid]]
) {
    const uint v = i < pc.VertexCount ? WorkGroupElement(b,pc.Vertices,i) : InvalidOffset;
    if (v == InvalidOffset) return;
    const CornerClassificationContext ctx{b,pc,{b,pc.Connectivity,pc.FaceCount}};
    const auto fan = ctx.Conn.Incoming(v);
    const bool touched = (ctx.Flags(fan)&5u) != 0u;
    for (uint n = 0u; n < fan.y; ++n) {
        const uint h = ctx.Conn.FanCorner(fan.x+n);
        const uint block = h/256u;
        const bool needed = touched && !ctx.Flat(h);
        if (needed) MarkWork(b,pc.NeededBlocks,block);
        if (needed || ctx.Payload(h) != InvalidOffset) MarkWork(b,pc.DirtyBlocks,block);
    }
}

kernel void CornerClassificationWrite(
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant CornerClassificationPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    uint i [[thread_position_in_grid]]
) {
    const uint v = i < pc.VertexCount ? WorkGroupElement(b,pc.Vertices,i) : InvalidOffset;
    if (v == InvalidOffset) return;
    const CornerClassificationContext ctx{b,pc,{b,pc.Connectivity,pc.FaceCount}};
    const auto fan = ctx.Conn.Incoming(v);
    for (uint n = 0u; n < fan.y; ++n) ctx.Write(ctx.Conn.FanCorner(fan.x+n),InvalidOffset);
    if (!(ctx.Flags(fan)&5u)) return;
    // A single vertex owns these writes. Incoming lists are sorted, so the
    // first unlabelled smooth corner is its component's canonical minimum.
    for (uint n = 0u; n < fan.y; ++n) {
        const uint h = ctx.Conn.FanCorner(fan.x+n);
        if (!ctx.Flat(h) && ctx.Root(h) == InvalidOffset) ctx.Sector(h,fan.y);
    }
}

kernel void CornerClassificationBlocks(
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant CornerClassificationPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]
) {
    const uint id = WorkGroupElement(b,pc.DirtyBlocks,group), h = id*256u+lane;
    const auto block = BindlessBuffer(MeshElementBlock,b.Buffer,pc.CornerBlocksSlot)[id];
    const CornerClassificationContext ctx{b,pc,{b,pc.Connectivity,pc.FaceCount}};
    uint flags = 0u;
    if (block.Owner == pc.CornerOwner && (block.Live[lane/32u] & (1u<<(lane%32u)))) {
        const uint root = ctx.Root(h);
        flags = uint(root != InvalidOffset) | (uint(root == h)<<1u);
    }
    flags = simd_or(flags);
    if (lane%32u == 0u) atomic_fetch_or_explicit(ctx.State()+pc.StatusOffset+group,flags,memory_order_relaxed);
}
