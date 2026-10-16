#include "NormalSectorTraversal.metal"
#include "ElementWorkShared.metal"
#include "gpu/CornerClassificationPushConstants.h"

// A lane's job and the vertex it classifies, InvalidOffset when the lane names none.
struct CornerClassificationContext {
    device const BindlessSet &B;
    constant CornerClassificationPushConstants &Pc;
    CornerClassificationJob Job;
    ConnectivityView Conn;
    uint Vertex;
    device atomic_uint *State() const { return BindlessBufferMutable(atomic_uint,B.Buffer,Pc.StorageSlot)+Pc.ScratchOffset+Job.State; }
    NormalSectorTraversal Sectors() const {
        return {Conn, BindlessBuffer(uchar,B.Buffer,Pc.FaceSharpnessSlot), BindlessBuffer(uchar,B.Buffer,Pc.EdgeSharpnessSlot)};
    }
    bool Flat(uint h) const { return Sectors().Flat(h); }
    bool Sharp(uint h) const { return Sectors().Sharp(h); }
    bool HasRoots() const { return Job.CornerSectors.Slot != InvalidSlot; }
    uint Root(uint h) const { return HasRoots() ? BindlessBuffer(uint,B.Buffer,Job.CornerSectors.Slot)[Job.CornerSectors.Offset+h] : InvalidOffset; }
    void Write(uint h, uint root) const {
        if (HasRoots()) BindlessBufferMutable(uint,B.Buffer,Job.CornerSectors.Slot)[Job.CornerSectors.Offset+h] = root;
    }
    uint Flags(uint2 fan) const {
        uint flags = 0u;
        for (uint i = 0u; i < fan.y; ++i) {
            const uint2 item = Conn.FanItem(fan.x+i);
            if (item.y==InvalidOffset) continue;
            const uint h=item.x;
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

inline CornerClassificationContext ClassificationLane(device const BindlessSet &b, constant CornerClassificationPushConstants &pc, uint group, uint lane) {
    device const uint *storage = BindlessBuffer(uint,b.Buffer,pc.StorageSlot);
    const uint2 tile = reinterpret_cast<device const uint2 *>(storage+pc.TileMapOffset)[pc.FirstTile+group];
    const CornerClassificationJob job = reinterpret_cast<device const CornerClassificationJob *>(storage+pc.JobsOffset)[tile.x];
    uint v;
    if (job.Vertices.Storage.Slot != InvalidSlot) {
        const uint i = tile.y*256u+lane;
        v = i < job.VertexCount ? WorkGroupElement(b,job.Vertices,i) : InvalidOffset;
    } else {
        const MeshElementBlock block = BindlessBuffer(MeshElementBlock,b.Buffer,pc.VertexBlocksSlot)[tile.y];
        v = block.Live[lane/32u] & (1u<<(lane%32u)) ? block.Local*256u+lane : InvalidOffset;
    }
    return {b,pc,job,{b,job.Connectivity,job.FaceCount},v};
}

kernel void CornerClassificationCount(
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant CornerClassificationPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]
) {
    const auto ctx = ClassificationLane(b,pc,group,lane);
    if (ctx.Vertex == InvalidOffset) return;
    const auto fan = ctx.Conn.Incoming(ctx.Vertex);
    atomic_fetch_add_explicit(ctx.State(),fan.y,memory_order_relaxed);
    atomic_fetch_or_explicit(ctx.State()+1u,ctx.Flags(fan),memory_order_relaxed);
}

kernel void CornerClassificationPlan(
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant CornerClassificationPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]
) {
    const auto ctx = ClassificationLane(b,pc,group,lane);
    if (ctx.Vertex == InvalidOffset) return;
    const auto fan = ctx.Conn.Incoming(ctx.Vertex);
    const bool touched = (ctx.Flags(fan)&5u) != 0u;
    for (uint n = 0u; n < fan.y; ++n) {
        const uint h = ctx.Conn.FanCorner(fan.x+n);
        const uint block = ctx.Conn.HalfedgeAddress(h)/256u;
        const bool needed = touched && ctx.Conn.HalfedgeFace(h)!=InvalidOffset && !ctx.Flat(h);
        if (needed) MarkWork(b,ctx.Job.NeededBlocks,block);
        if (needed || ctx.HasRoots()) MarkWork(b,ctx.Job.DirtyBlocks,block);
    }
}

kernel void CornerClassificationWrite(
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant CornerClassificationPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]
) {
    const auto ctx = ClassificationLane(b,pc,group,lane);
    if (ctx.Vertex == InvalidOffset) return;
    const auto fan = ctx.Conn.Incoming(ctx.Vertex);
    for (uint n = 0u; n < fan.y; ++n) ctx.Write(ctx.Conn.FanCorner(fan.x+n),InvalidOffset);
    if (!(ctx.Flags(fan)&5u)) return;
    // A single vertex owns these writes.
    // Incoming lists are sorted, so the first unlabelled smooth corner is its component's canonical minimum.
    for (uint n = 0u; n < fan.y; ++n) {
        const uint h = ctx.Conn.FanCorner(fan.x+n);
        if (ctx.Conn.HalfedgeFace(h)!=InvalidOffset && !ctx.Flat(h) && ctx.Root(h) == InvalidOffset) ctx.Sector(h,fan.y);
    }
}
