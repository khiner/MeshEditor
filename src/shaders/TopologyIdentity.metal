#include "MeshTopologyContext.metal"
#include "gpu/TopologyIdentityPushConstants.h"

// An output map, indexed by compact output ordinal.
inline device uint *IdentityOutput(device const BindlessSet &b, SlotOffset map) { return BindlessBufferMutable(uint,b.Buffer,map.Slot)+map.Offset; }

// The operator's scanned output count of a quantity, which the host reads only after this submit.
inline uint IdentityOutputCount(TopoContext ctx, MeshTopologyJob job, uint quantity) {
    return ctx.Counts(job, quantity)[job.CountEntries - 1u];
}

kernel void TopologyIdentityInit(
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant TopologyIdentityPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    uint i [[thread_position_in_grid]]
) {
    const TopoContext ctx{b,pc.Topology};
    if (i < IdentityOutputCount(ctx,pc.Job,TopoCountVertices)) IdentityOutput(b,pc.Outputs[0])[i] = InvalidOffset;
    if (i < IdentityOutputCount(ctx,pc.Job,TopoCountFaces)) IdentityOutput(b,pc.Outputs[1])[i] = InvalidOffset;
}

// A source vertex's own output precedes its copies.
// A source face's first emitted face keeps its identity.
// Edge/vertex-generated faces are new.
kernel void TopologyIdentityRetain(
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant TopologyIdentityPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    uint i [[thread_position_in_grid]]
) {
    const TopoContext ctx{b,pc.Topology};
    const auto job = pc.Job;
    if (i < job.SrcVertexCount) {
        device const uint *offsets = ctx.Counts(job,TopoCountVertices);
        if (offsets[i] != offsets[i+1u] && TopoVertexKept(ctx,job,i)) {
            IdentityOutput(b,pc.Outputs[0])[offsets[i]] = ctx.SrcVertexDomain(job).Handle(i);
        } else MarkWork(b,pc.RetiredElements[0],ctx.SrcVertexDomain(job).Handle(i));
    }
    if (i < job.SrcFaceCount) {
        const uint entry = ctx.FaceEntry(job,i);
        device const uint *offsets = ctx.Counts(job,TopoCountFaces);
        if (offsets[entry] != offsets[entry+1u]) {
            IdentityOutput(b,pc.Outputs[1])[offsets[entry]] = ctx.SrcFaceDomain(job).Handle(i);
        } else MarkWork(b,pc.RetiredElements[1],ctx.SrcFaceDomain(job).Handle(i));
    }
}

// One thread per old corner also visits its fan triangle, when one exists.
// Capture ownership before face starts/ranges are overwritten by emission.
// A line corner derives no triangle.
kernel void TopologyIdentityReplaced(
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant TopologyIdentityPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    uint i [[thread_position_in_grid]]
) {
    const auto job = pc.Job;
    if (i >= job.SrcHalfedgeCount) return;
    const TopoContext ctx{b,pc.Topology};
    const auto fail = [&] { atomic_store_explicit(BindlessBufferMutable(atomic_uint,b.Buffer,pc.Error.Slot)+pc.Error.Offset,1u,memory_order_relaxed); };
    const uint h = ctx.SrcHalfedgeDomain(job).Handle(i);
    if (h >= pc.ReplacedElements[0].Count) { fail(); return; }
    if (TopoLineCore(job)) {
        MarkWork(b,pc.ReplacedElements[0],h);
        return;
    }
    const uint face = ctx.SrcFaceOf(job,h);
    if (face >= job.SrcFaceCount) { fail(); return; }
    const uint2 range = ctx.SrcFaceRange(job,face);
    if (range.y < range.x || range.y-range.x < 3u || h < range.x || h >= range.y) { fail(); return; }
    MarkWork(b,pc.ReplacedElements[0],h);
    if (h-range.x < range.y-range.x-2u) {
        const uint first = BindlessBuffer(uint,b.ObjectIdBuffer,pc.Topology.Source.FaceTriangleStartSlot)[ctx.SrcFaceDomain(job).Handle(face)];
        const ulong triangle = ulong(first)+h-range.x;
        if (triangle >= pc.ReplacedElements[1].Count) { fail(); return; }
        MarkWork(b,pc.ReplacedElements[1],uint(triangle));
    }
}

kernel void TopologyIdentityNew(
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant TopologyIdentityPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    uint3 tid [[thread_position_in_grid]]
) {
    const uint d = tid.z, i = tid.x;
    const TopoContext ctx{b,pc.Topology};
    if (i < IdentityOutputCount(ctx,pc.Job,d == 0u ? TopoCountVertices : TopoCountFaces) &&
        IdentityOutput(b,pc.Outputs[d])[i] == InvalidOffset)
        MarkWork(b,pc.NewElements[d],i);
}

// The i-th new output takes the i-th inserted handle.
kernel void TopologyIdentityAssign(
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant TopologyIdentityPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    uint3 tid [[thread_position_in_grid]]
) {
    const uint d = tid.z, i = tid.x;
    const ElementHandleRange inserted = pc.Inserted[d];
    if (i >= inserted.Count) return;
    const uint ordinal = WorkGroupElement(b,pc.NewElements[d],i);
    if (ordinal == InvalidOffset) {
        atomic_store_explicit(BindlessBufferMutable(atomic_uint,b.Buffer,pc.Error.Slot)+pc.Error.Offset,1u,memory_order_relaxed);
        return;
    }
    IdentityOutput(b,pc.Outputs[d])[ordinal] = inserted.Handles.Slot == InvalidSlot ? inserted.First+i :
        BindlessBuffer(uint,b.Buffer,inserted.Handles.Slot)[inserted.Handles.Offset+i];
}
