#include "MeshTopologyContext.metal"
#include "gpu/TopologyIdentityJob.h"

inline device uint *IdentityOutput(device const BindlessSet &b, SlotOffset map) { return BindlessBufferMutable(uint,b.Buffer,map.Slot)+map.Offset; }
// The identity passes read the batch storage their jobs locate, with every other topology constant at its default.
constant MeshTopologyPushConstants IdentityTopology{};

// The operator's scanned output count of a quantity, which the host reads only after this submit.
inline uint IdentityOutputCount(TopoContext ctx, device const MeshTopologyJob &job, uint quantity) {
    return ctx.Counts(job, quantity)[job.CountEntries - 1u];
}
inline device const MeshTopologyJob &IdentityTopologyJob(TopoContext ctx, constant TopologyIdentityJob &identity) {
    return *reinterpret_cast<device const MeshTopologyJob *>(ctx.Storage()+identity.Job);
}
// Declares the identity job of the kernel's threadgroup, its operator's context and job, and the thread's index `i` within the job.
#define IDENTITY_THREAD() JOB_THREAD(identity, i); const TopoContext ctx{b, IdentityTopology, identity.Scratch}; device const auto &job = IdentityTopologyJob(ctx, identity)

kernel void TopologyIdentityInit(JOB_TABLE_ARGS(TopologyIdentityJob)) {
    IDENTITY_THREAD();
    if (i < IdentityOutputCount(ctx,job,TopoCountVertices)) IdentityOutput(b,identity.Outputs[0])[i] = InvalidOffset;
    if (i < IdentityOutputCount(ctx,job,TopoCountFaces)) IdentityOutput(b,identity.Outputs[1])[i] = InvalidOffset;
}

// A source vertex's own output precedes its copies.
// A source face's first emitted face keeps its identity.
// Edge/vertex-generated faces are new.
kernel void TopologyIdentityRetain(JOB_TABLE_ARGS(TopologyIdentityJob)) {
    IDENTITY_THREAD();
    if (i < job.SrcVertexCount) {
        device const uint *offsets = ctx.Counts(job,TopoCountVertices);
        if (offsets[i] != offsets[i+1u] && TopoVertexKept(ctx,job,i)) {
            IdentityOutput(b,identity.Outputs[0])[offsets[i]] = ctx.SrcVertexDomain(job).Handle(i);
        } else MarkWork(b,identity.RetiredElements[0],ctx.SrcVertexDomain(job).Handle(i));
    }
    if (i < job.SrcFaceCount) {
        const uint entry = ctx.FaceEntry(job,i);
        device const uint *offsets = ctx.Counts(job,TopoCountFaces);
        if (offsets[entry] != offsets[entry+1u]) {
            IdentityOutput(b,identity.Outputs[1])[offsets[entry]] = ctx.SrcFaceDomain(job).Handle(i);
        } else MarkWork(b,identity.RetiredElements[1],ctx.SrcFaceDomain(job).Handle(i));
    }
}

// One thread per old corner also visits its fan triangle, when one exists.
// Capture ownership before face starts/ranges are overwritten by emission.
// A line corner derives no triangle.
kernel void TopologyIdentityReplaced(JOB_TABLE_ARGS(TopologyIdentityJob)) {
    IDENTITY_THREAD();
    if (i >= job.SrcHalfedgeCount) return;
    const auto fail = [&] { atomic_store_explicit(BindlessBufferMutable(atomic_uint,b.Buffer,identity.ErrorSlot),1u,memory_order_relaxed); };
    const uint h = ctx.SrcHalfedgeDomain(job).Handle(i);
    if (h >= identity.ReplacedElements[0].Count) { fail(); return; }
    if (ctx.Src(job).HalfedgeFace(h) == InvalidOffset) {
        MarkWork(b,identity.ReplacedElements[0],h);
        return;
    }
    const uint face = ctx.SrcFaceOf(job,h);
    if (face >= job.SrcFaceCount) { fail(); return; }
    const uint2 range = ctx.SrcFaceRange(job,face);
    if (range.y < range.x || range.y-range.x < 3u || h < range.x || h >= range.y) { fail(); return; }
    MarkWork(b,identity.ReplacedElements[0],h);
    if (h-range.x < range.y-range.x-2u) {
        const uint first = BindlessBuffer(uint,b.ObjectIdBuffer,identity.FaceTriangleStartSlot)[ctx.SrcAddress(job,2u,ctx.SrcFaceDomain(job).Handle(face))];
        const ulong triangle = ulong(first)+h-range.x;
        if (triangle >= identity.ReplacedElements[1].Count) { fail(); return; }
        MarkWork(b,identity.ReplacedElements[1],uint(triangle));
    }
}

// The second layer of threadgroups covers faces.
kernel void TopologyIdentityNew(JOB_TABLE_LAYERED_ARGS(TopologyIdentityJob)) {
    IDENTITY_THREAD();
    const uint d = group.z;
    if (i < IdentityOutputCount(ctx,job,d == 0u ? TopoCountVertices : TopoCountFaces) &&
        IdentityOutput(b,identity.Outputs[d])[i] == InvalidOffset)
        MarkWork(b,identity.NewElements[d],i);
}

kernel void TopologyIdentityAssign(JOB_TABLE_LAYERED_ARGS(TopologyIdentityJob)) {
    JOB_THREAD(identity, i);
    const uint d = group.z;
    const ElementHandleRange inserted = identity.Inserted[d];
    if (i >= inserted.Count) return;
    const uint ordinal = WorkGroupElement(b,identity.NewElements[d],i);
    if (ordinal == InvalidOffset) {
        atomic_store_explicit(BindlessBufferMutable(atomic_uint,b.Buffer,identity.ErrorSlot),1u,memory_order_relaxed);
        return;
    }
    IdentityOutput(b,identity.Outputs[d])[ordinal] = inserted.Handles.Slot == InvalidSlot ? inserted.First+i :
        BindlessBuffer(uint,b.Buffer,inserted.Handles.Slot)[inserted.Handles.Offset+i];
}
