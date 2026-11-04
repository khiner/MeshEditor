#pragma once
#include "MeshTopologyContext.metal"

// Planning pushes each source corner once, and a repeated target vertex pops a simple loop off the stack.
// The hash table keeps stale stack indices and checks them against the current stack, so a pop needs no tombstones or table scan.
struct TopoWeldLoops {
    device uint *Keys, *Indices;
    device packed_uint2 *Stack, *Corners; // Attribute corner, incoming edge corner.
    device uint *Starts, *Lengths;
};

inline TopoWeldLoops TopoWeldPlan(TopoContext ctx, MeshTopologyJob job, uint f) {
    const uint2 range=ctx.SrcFaceRange(job,f);
    const uint n=range.y-range.x;
    device uint *base=ctx.Scratch()+job.FaceLoopOffset+14u*ctx.SrcHalfedgeDomain(job).Index(range.x);
    return {base,base+4u*n,reinterpret_cast<device packed_uint2 *>(base+8u*n),
        reinterpret_cast<device packed_uint2 *>(base+10u*n),base+12u*n,base+13u*n};
}

inline void TopoPlanWeld(TopoContext ctx, MeshTopologyJob job, uint f) {
    const uint2 range=ctx.SrcFaceRange(job,f);
    const uint n=range.y-range.x,capacity=1u<<(32u-clz(2u*n-1u));
    const auto plan=TopoWeldPlan(ctx,job,f);
    const auto source=ctx.SrcCorners(job);
    const auto target=[&](uint h) { return ctx.VertexTargets(job)[source[h]]; };
    for (uint k=0u;k<capacity;++k) plan.Keys[k]=InvalidOffset;
    uint top=0u,faces=0u,emitted=0u;
    for (uint i=0u;i<=n;++i) {
        const uint h=range.x+(i==n ? 0u : i),v=target(h);
        uint slot=WorkHash(v,capacity);
        while (plan.Keys[slot]!=InvalidOffset && plan.Keys[slot]!=v) slot=(slot+1u)&(capacity-1u);
        const uint previous=plan.Keys[slot]==v ? plan.Indices[slot] : InvalidOffset;
        if (previous<top && target(plan.Stack[previous].x)==v) {
            const uint length=top-previous;
            if (length>=3u) {
                plan.Starts[faces]=emitted; plan.Lengths[faces++]=length;
                plan.Corners[emitted++]=packed_uint2(plan.Stack[previous].x,h);
                ctx.FlagHalfedges(job)[h]|=TopoSurfaceEdge;
                for (uint k=previous+1u;k<top;++k) {
                    const auto corner=plan.Stack[k];
                    plan.Corners[emitted++]=corner;
                    ctx.FlagHalfedges(job)[corner.y]|=TopoSurfaceEdge;
                }
            }
            top=previous+1u;
            // The remainder keeps its incoming edge and takes the attributes of the corner whose outgoing edge it now follows.
            plan.Stack[previous].x=h;
        } else {
            plan.Keys[slot]=v; plan.Indices[slot]=top;
            plan.Stack[top++]=packed_uint2(h,h);
        }
    }
    ctx.FlagFaces(job)[f]=faces;
}

// After planning, Stack holds each polygon's lowest corner and hash, and Keys holds its changed, selected and keep flags.
struct TopoWeldPolygon {
    TopoWeldLoops Plan;
    uint Index, Size;
};

inline TopoWeldPolygon TopoWeldPolygonAt(TopoContext ctx, MeshTopologyJob job, uint hi) {
    const uint h=ctx.SrcHalfedgeDomain(job).Handle(hi),f=ctx.SrcFaceOf(job,h);
    const auto plan=TopoWeldPlan(ctx,job,f);
    const uint2 range=ctx.SrcFaceRange(job,f);
    return {plan,plan.Indices[h-range.x],range.y-range.x};
}

inline uint TopoWeldVertex(TopoContext ctx, MeshTopologyJob job, TopoWeldPolygon p, uint k) {
    return ctx.VertexTargets(job)[ctx.SrcCorners(job)[p.Plan.Corners[p.Plan.Starts[p.Index]+k].x]];
}

// Whether the polygon winds forward from its lowest vertex, which coincident polygons share when they wind the same way.
inline bool TopoWeldForward(TopoContext ctx, MeshTopologyJob job, TopoWeldPolygon p) {
    const uint n=p.Plan.Lengths[p.Index],first=p.Plan.Stack[p.Index].x;
    return TopoWeldVertex(ctx,job,p,(first+1u)%n)<TopoWeldVertex(ctx,job,p,(first+n-1u)%n);
}
inline uint TopoWeldCanonicalVertex(TopoContext ctx, MeshTopologyJob job, TopoWeldPolygon p, uint k) {
    const uint n=p.Plan.Lengths[p.Index],first=p.Plan.Stack[p.Index].x;
    return TopoWeldVertex(ctx,job,p,(first+(TopoWeldForward(ctx,job,p) ? k : n-k))%n);
}

inline void TopoWeldKeys(TopoContext ctx, MeshTopologyJob job, uint f) {
    const auto plan=TopoWeldPlan(ctx,job,f);
    const uint2 range=ctx.SrcFaceRange(job,f);
    for (uint i=0u;i<ctx.FlagFaces(job)[f];++i) {
        const TopoWeldPolygon p{plan,i,range.y-range.x};
        const uint n=plan.Lengths[i];
        uint first=0u,changed=n!=p.Size;
        for (uint k=0u;k<n;++k) {
            if (TopoWeldVertex(ctx,job,p,k)<TopoWeldVertex(ctx,job,p,first)) first=k;
            changed|=TopoWeldVertex(ctx,job,p,k)!=ctx.SrcCorners(job)[plan.Corners[plan.Starts[i]+k].x];
        }
        plan.Stack[i].x=first;
        uint hash=n;
        for (uint k=0u;k<n;++k) hash=(hash^TopoWeldCanonicalVertex(ctx,job,p,k))*16777619u;
        plan.Stack[i].y=hash;
        plan.Keys[i]=changed;
        plan.Keys[p.Size+i]=ctx.SrcSelectedFace(job,f);
        plan.Keys[2u*p.Size+i]=1u;
        plan.Indices[plan.Corners[plan.Starts[i]].x-range.x]=i;
    }
}

inline bool TopoWeldEqual(TopoContext ctx, MeshTopologyJob job, TopoWeldPolygon a, TopoWeldPolygon b) {
    const uint n=a.Plan.Lengths[a.Index];
    if (n!=b.Plan.Lengths[b.Index] || a.Plan.Stack[a.Index].y!=b.Plan.Stack[b.Index].y) return false;
    for (uint k=0u;k<n;++k)
        if (TopoWeldCanonicalVertex(ctx,job,a,k)!=TopoWeldCanonicalVertex(ctx,job,b,k)) return false;
    return true;
}

inline void TopoWeldInsert(TopoContext ctx, MeshTopologyJob job, uint f) {
    const auto plan=TopoWeldPlan(ctx,job,f);
    const uint2 range=ctx.SrcFaceRange(job,f);
    device atomic_uint *table=AtomicWord(ctx.Table(job));
    for (uint i=0u;i<ctx.FlagFaces(job)[f];++i) {
        const TopoWeldPolygon p{plan,i,range.y-range.x};
        const uint hi=ctx.SrcHalfedgeDomain(job).Index(plan.Corners[plan.Starts[i]].x);
        uint slot=plan.Stack[i].y&job.TableMask;
        for (uint probe=0u;probe<=job.TableMask;) {
            uint occupant=InvalidOffset;
            if (atomic_compare_exchange_weak_explicit(&table[slot],&occupant,hi,memory_order_relaxed,memory_order_relaxed)) break;
            if (occupant==InvalidOffset) continue;
            const auto other=TopoWeldPolygonAt(ctx,job,occupant);
            if (TopoWeldEqual(ctx,job,p,other)) {
                const uint priority=plan.Keys[i],other_priority=other.Plan.Keys[other.Index];
                if (other_priority<priority || (other_priority==priority && occupant<=hi)) break;
                if (atomic_compare_exchange_weak_explicit(&table[slot],&occupant,hi,memory_order_relaxed,memory_order_relaxed)) break;
                continue;
            }
            slot=(slot+1u)&job.TableMask; ++probe;
        }
    }
}

inline void TopoWeldDeduplicate(TopoContext ctx, MeshTopologyJob job, uint f) {
    const auto plan=TopoWeldPlan(ctx,job,f);
    const uint2 range=ctx.SrcFaceRange(job,f);
    for (uint h=range.x;h<range.y;++h) ctx.FlagHalfedges(job)[h]&=~TopoSurfaceEdge;
    for (uint i=0u;i<ctx.FlagFaces(job)[f];++i) {
        const TopoWeldPolygon p{plan,i,range.y-range.x};
        const uint hi=ctx.SrcHalfedgeDomain(job).Index(plan.Corners[plan.Starts[i]].x);
        uint slot=plan.Stack[i].y&job.TableMask;
        for (uint probe=0u;probe<=job.TableMask;++probe,slot=(slot+1u)&job.TableMask) {
            const uint occupant=ctx.Table(job)[slot];
            if (occupant==InvalidOffset) break;
            const auto other=TopoWeldPolygonAt(ctx,job,occupant);
            if (!TopoWeldEqual(ctx,job,p,other)) continue;
            // Coincident faces winding opposite ways are flaps that drop in pairs when the job drops flaps.
            if (occupant!=hi && (job.Flags & TopologyFlagDropFlaps) && TopoWeldForward(ctx,job,p)!=TopoWeldForward(ctx,job,other)) {
                plan.Keys[2u*p.Size+i]=0u;
                atomic_store_explicit(AtomicWord(other.Plan.Keys+2u*other.Size+other.Index),0u,memory_order_relaxed);
            } else if (occupant!=hi && plan.Keys[i]) {
                plan.Keys[2u*p.Size+i]=0u;
                if (ctx.SrcSelectedFace(job,f))
                    atomic_fetch_or_explicit(AtomicWord(other.Plan.Keys+other.Size+other.Index),1u,memory_order_relaxed);
            }
            break;
        }
        if (plan.Keys[2u*p.Size+i]) for (uint k=0u;k<plan.Lengths[i];++k)
            ctx.FlagHalfedges(job)[plan.Corners[plan.Starts[i]+k].y]|=TopoSurfaceEdge;
    }
}

inline uint3 TopoWeldCounts(TopoContext ctx, MeshTopologyJob job, uint f) {
    const auto plan=TopoWeldPlan(ctx,job,f);
    const uint2 range=ctx.SrcFaceRange(job,f);
    uint3 counts=uint3(0u);
    for (uint i=0u;i<ctx.FlagFaces(job)[f];++i) if (plan.Keys[2u*(range.y-range.x)+i]) {
        ++counts.y; counts.z+=plan.Lengths[i];
    }
    return counts;
}

inline void TopoEmitWeld(TopoContext ctx, MeshTopologyJob job, uint f, uint fd, uint base) {
    const auto plan=TopoWeldPlan(ctx,job,f);
    const uint2 range=ctx.SrcFaceRange(job,f);
    const uint n=range.y-range.x;
    const auto source=ctx.SrcCorners(job);
    const auto vertices=ctx.Counts(job,TopoCountVertices);
    for (uint polygon=0u;polygon<ctx.FlagFaces(job)[f];++polygon) {
        if (!plan.Keys[2u*n+polygon]) continue;
        const uint count=plan.Lengths[polygon],first=plan.Starts[polygon];
        for (uint k=0u;k<count;++k) {
            const uint2 corner=uint2(plan.Corners[first+k]);
            const uint v=ctx.VertexTargets(job)[source[corner.x]];
            ctx.WriteCorner(job,base+k,vertices[v],corner.x,corner.x,0.f,corner.y,ctx.SrcSelectedEdge(job,ctx.SrcEdge(job,corner.y)));
        }
        TopoEmitFace(ctx,job,fd++,base,count,f,plan.Keys[n+polygon]!=0u);
        base+=count;
    }
}
