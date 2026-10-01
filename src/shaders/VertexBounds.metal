#ifndef VERTEXBOUNDS_MSL
#define VERTEXBOUNDS_MSL
#include "Bindless.metal"
#include "ElementWorkShared.metal"
#include "gpu/VertexBounds.h"
#include "gpu/BoundsReducePushConstants.h"

// The value index of a bounds record in the namespace at `root`, or InvalidOffset when the namespace holds none.
inline uint VertexBoundsFind(device const BindlessSet &b, uint nodes_slot, uint members_slot, uint root, uint record) {
    uint id=root;
    if (id==InvalidOffset) return id;
    device const VertexBoundsMapNode *nodes=BindlessBuffer(VertexBoundsMapNode,b.Buffer,nodes_slot);
    for (uint depth=3u; depth--;) {
        const uint child=nodes[id].Children[(record>>(5u+depth*7u))&127u];
        if (!child) return InvalidOffset;
        id=child-1u;
    }
    const uint members=BindlessBuffer(uint,b.Buffer,members_slot)[id];
    return members & (1u<<(record%32u)) ? id*32u+record%32u : InvalidOffset;
}
inline uint VertexBoundsIndex(device const BindlessSet &b, constant BoundsReducePushConstants &pc, BoundsEntry entry, uint level, uint key) {
    return VertexBoundsFind(b,pc.NodesSlot,pc.MembersSlot,entry.BoundsNamespace,VertexBoundsKey(level,key));
}
inline bool BoundsVertexLive(device const BindlessSet &b, BoundsEntry entry, uint block, uint lane) {
    const MeshElementBlock membership=BindlessBuffer(MeshElementBlock,b.Buffer,entry.VertexBlocksSlot)[block];
    return membership.Owner == entry.VertexOwner && (membership.Live[lane/32u] & (1u << (lane%32u))) != 0u;
}
inline uint2 VertexBoundsTile(device const BindlessSet &b, constant BoundsReducePushConstants &pc, uint group) {
    return pc.Work.Storage.Slot == InvalidSlot ?
        uint2(BindlessBuffer(packed_uint2,b.Buffer,pc.TileMapSlot)[pc.FirstTile+group]) :
        uint2(pc.EntryIndex,WorkGroupElement(b,pc.Work,group));
}
#endif
