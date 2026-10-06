#ifndef MESHTOPOLOGYLIMITED_MSL
#define MESHTOPOLOGYLIMITED_MSL

#include "MeshTopologyContext.metal"

// An edge's two face users can disagree in any UV set or material, or carry
// an explicit sharp mark. These are canonical arena reads over affected edges.
inline bool TopoLimitedDelimiter(TopoContext ctx,MeshTopologyJob job,uint h,uint opposite) {
    if ((job.Flags&TopologyFlagDelimitSharp) && ctx.SrcEdgeSharpness(job)[ctx.SrcEdge(job,h)]) return true;
    // Primitive indices are the mesh's material slots, including distinct
    // slots that currently reference the same material.
    if ((job.Flags&TopologyFlagDelimitMaterial) &&
        ctx.SrcElementPrimitive(job,ctx.SrcFaceOf(job,h))!=ctx.SrcElementPrimitive(job,ctx.SrcFaceOf(job,opposite))) return true;
    if (job.Flags&TopologyFlagDelimitUV) {
        const uint previous=ctx.SrcPrev(job,h),other_previous=ctx.SrcPrev(job,opposite);
        for (uint set=0u;set<4u;++set) if (job.CornerAttributes&(MeshAttributeBit_TexCoord0<<set))
            if (any(ctx.SrcCornerUv(job,set,h)!=ctx.SrcCornerUv(job,set,other_previous)) ||
                any(ctx.SrcCornerUv(job,set,previous)!=ctx.SrcCornerUv(job,set,opposite))) return true;
    }
    return false;
}

struct TopoLimitedWork {
    device packed_uint2 *Neighbors;
    device uint *Labels, *Queue;
};
inline TopoLimitedWork TopoLimited(TopoContext ctx,MeshTopologyJob job) {
    device uint *base=ctx.Scratch()+job.FaceLoopOffset+(job.SrcFaceCount ? job.SrcHalfedgeCount : 0u);
    return {reinterpret_cast<device packed_uint2 *>(base),base+2u*job.SrcVertexCount,base+3u*job.SrcVertexCount};
}

// Build chains of selected degree-two vertices. Their endpoints, branches and
// unselected vertices stay outside the chains and are never written by them.
inline void TopoLimitedPrepare(TopoContext ctx,MeshTopologyJob job,uint v) {
    const auto work=TopoLimited(ctx,job);
    work.Labels[v]=InvalidOffset;
    if ((job.Param0<=0.f && !(job.Flags&TopologyFlagAllBoundaries)) || !ctx.SrcSelectedVertex(job,v) ||
        ctx.VertexEdgeTotal(job)[v]-ctx.VertexEdgeDissolved(job)[v]!=2u) return;
    const auto src=ctx.Src(job);
    uint2 neighbors=uint2(InvalidOffset);
    uint first_edge=InvalidOffset,count=0u;
    for (const auto item:src.Fan(ctx.SrcVertexDomain(job).Handle(v))) {
        if ((job.Flags&TopologyFlagFaceSelection) && item.y!=InvalidOffset &&
            !ctx.SrcSelectedFace(job,ctx.SrcFaceDomain(job).Index(item.y))) return;
        const uint2 sides=uint2(item.x,item.y==InvalidOffset ? InvalidOffset : src.Next(item.x,item.y));
        for (uint k=0u;k<2u;++k) {
            const uint h=sides[k];
            if (h==InvalidOffset || TopoEdgeDissolved(ctx,job,h)) continue;
            const uint e=src.Edge(h);
            if (e==first_edge) continue;
            const auto corners=ctx.SrcCorners(job);
            const uint a=corners[h],b=corners[ctx.SrcPrev(job,h)];
            if (count<2u) neighbors[count++]=a==v ? b : a;
            first_edge=e;
        }
    }
    if (count!=2u || any(neighbors==uint2(InvalidOffset))) return;
    work.Neighbors[v]=packed_uint2(neighbors);
    work.Labels[v]=v;
}

inline void TopoLimitedJump(TopoContext ctx,MeshTopologyJob job,uint v) {
    const auto work=TopoLimited(ctx,job);
    device atomic_uint *labels=ctx.Atomic(work.Labels);
    const uint old=atomic_load_explicit(labels+v,memory_order_relaxed);
    if (old==InvalidOffset) return;
    const uint2 neighbors=uint2(work.Neighbors[v]);
    uint root=atomic_load_explicit(labels+old,memory_order_relaxed);
    root=min(root,atomic_load_explicit(labels+neighbors.x,memory_order_relaxed));
    root=min(root,atomic_load_explicit(labels+neighbors.y,memory_order_relaxed));
    if (atomic_fetch_min_explicit(labels+v,root,memory_order_relaxed)>root) ctx.State(job)[1]=1u;
}

// One GPU thread owns each chain. A removal changes only its two neighbors,
// which re-enter the queue if needed. Thus there are at most N+2N queue visits;
// a curved chain cannot disappear merely because all original bends were small.
inline void TopoLimitedSimplify(TopoContext ctx,MeshTopologyJob job,uint root) {
    if (!(ctx.State(job)[0]&TopoLabelsConverged)) return;
    const auto work=TopoLimited(ctx,job);
    if (work.Labels[root]!=root) return;
    const auto flags=ctx.FlagVertices(job);
    uint head=InvalidOffset,tail=InvalidOffset;
    const auto enqueue=[&](uint v) {
        if (work.Labels[v]==InvalidOffset || (flags[v]&(TopoLimitedQueued|TopoDissolvable))) return;
        flags[v]|=TopoLimitedQueued;
        work.Queue[v]=InvalidOffset;
        if (tail!=InvalidOffset) work.Queue[tail]=v;
        else head=v;
        tail=v;
    };
    enqueue(root);
    // Discover the component before changing any links, including closed wires.
    for (uint v=head;v!=InvalidOffset;v=work.Queue[v]) {
        const uint2 neighbors=uint2(work.Neighbors[v]);
        enqueue(neighbors.x); enqueue(neighbors.y);
    }
    const float limit=-cos(job.Param0);
    while (head!=InvalidOffset) {
        const uint v=head;
        head=work.Queue[v]; if (head==InvalidOffset) tail=InvalidOffset;
        flags[v]&=~TopoLimitedQueued;
        const uint2 neighbors=uint2(work.Neighbors[v]);
        if (neighbors.x==neighbors.y || ctx.VertexEdgeTotal(job)[v]-ctx.VertexEdgeDissolved(job)[v]!=2u) continue;
        if (!(job.Flags&TopologyFlagAllBoundaries)) {
            const float3 p=ctx.SrcPosition(job,v),a=ctx.SrcPosition(job,neighbors.x)-p,b=ctx.SrcPosition(job,neighbors.y)-p;
            const float product=length_squared(a)*length_squared(b);
            if (product<=0.f || dot(a,b)>=limit*sqrt(product)) continue;
        }
        flags[v]|=TopoDissolvable;
        for (uint k=0u;k<2u;++k) {
            const uint n=neighbors[k],other=neighbors[1u-k];
            if (work.Labels[n]==InvalidOffset) continue;
            uint2 links=uint2(work.Neighbors[n]);
            if (links.x==v) links.x=other;
            if (links.y==v) links.y=other;
            work.Neighbors[n]=packed_uint2(links);
            enqueue(n);
        }
    }
}

#endif
