#ifndef MESHTOPOLOGY_MSL
#define MESHTOPOLOGY_MSL

// Runs one edit-mode topology operator from a source mesh and its selection into a new mesh.
// Every operator marks, counts, scans, scatters, and gathers.
// Source elements produce output vertices, faces, and corners at scanned offsets, and index maps carry every attribute domain into the output.
#include "MeshTopologyBevel.metal"
#include "MeshTopologyContext.metal"
#include "MeshTopologyFaces.metal"
#include "MeshTopologyLines.metal"
#include "MeshTopologyLimited.metal"
#include "MeshTopologySubdivide.metal"
#include "MeshTopologyWireframe.metal"
#include "MeshTopologyFaceSplit.metal"
#include "MeshTopologyConcave.metal"
#include "MeshTopologyWeld.metal"
#include "gpu/InsetVertexBasis.h"

// The vertices, faces, and corners a source face produces.
inline uint3 TopoFaceOutputs(TopoContext ctx, MeshTopologyJob job, uint f) {
    if (TopologyIsMerge(job.Op)) return TopoWeldCounts(ctx,job,f);
    const uint2 range = ctx.SrcFaceRange(job, f);
    const uint n = range.y - range.x;
    const bool selected = ctx.SrcSelectedFace(job, f);
    switch (TopologyBaseOp(job.Op)) {
        case MeshTopologyOp::ReplaceFaces: {
            const auto list=ctx.Lists(job);
            const auto record=list+list[2u+list[0]+f];
            uint cursor=2u,corners=0u;
            for (uint i=0u;i<record[1];++i) { const uint n=record[cursor++]; corners+=n; cursor+=2u*n; }
            return uint3(0u,record[1],corners);
        }
        case MeshTopologyOp::Wireframe: return TopoWireFaceCounts(ctx, job, f);
        case MeshTopologyOp::SplitNonplanarFaces: return TopoNonplanarCount(ctx, job, f);
        case MeshTopologyOp::SplitConcaveFaces: return TopoConcaveCount(ctx, job, f);
        case MeshTopologyOp::Triangulate:
            return selected && n > 3u ? uint3(0u, n - 2u, 3u * (n - 2u)) : uint3(0u, 1u, n);
        case MeshTopologyOp::TrisToQuads: {
            const uint partner = ctx.FaceLabels(job)[f];
            if (partner == InvalidOffset) return uint3(0u, 1u, n);
            return partner > f ? uint3(0u, 1u, 4u) : uint3(0u);
        }
        case MeshTopologyOp::Poke:
            return selected ? uint3(1u, n, 3u * n) : uint3(0u, 1u, n);
        case MeshTopologyOp::FlipNormals:
        case MeshTopologyOp::EdgeSplit:
        case MeshTopologyOp::AddPrimitives:
            return uint3(0u, 1u, n);
        case MeshTopologyOp::ExtrudeRegion:
            if (job.Op==MeshTopologyOp::ExtrudeRegion && selected) {
                const uint faces=uint(!ctx.DelOrig(job))+(TopoRegionHasSides(ctx,job) ? 1u : job.Steps);
                return uint3(0u,faces,n*faces);
            }
        case MeshTopologyOp::DuplicateGeometry:
            return selected && TopoRegionDuplicates(ctx, job) ? uint3(0u, 1u + job.Steps, n * (1u + job.Steps)) : uint3(0u, 1u, n);
        case MeshTopologyOp::ExtrudeFacesIndividual:
            return selected ? uint3(n, 1u + n, 5u * n) : uint3(0u, 1u, n);
        case MeshTopologyOp::Subdivide: {
            SubdivideEmitter emitter{ctx, job, f, false, 0u, 0u, 0u, 0u};
            const uint interior = TopoSubdivideFace(ctx, job, f, emitter);
            return uint3(interior, emitter.Faces, emitter.Corners);
        }
        case MeshTopologyOp::ConnectVertices:
        case MeshTopologyOp::RotateEdges: {
            const bool own_loop = job.Op == MeshTopologyOp::ConnectVertices || TopoDissolveOwnLoop(ctx, job, f);
            if (!own_loop && ctx.FaceLabels(job)[f] != f) return uint3(0u);
            const uint2 output = TopoConnectOutputs(TopoMeasureConnect(ctx, job, f, own_loop));
            return uint3(0u, output.x, output.y);
        }
        case MeshTopologyOp::BevelEdges:
        case MeshTopologyOp::BevelVertices: {
            uint corners = 0u;
            for (uint h = range.x; h < range.y; ++h) corners += TopoBevelCornerPoints(ctx, job, h).Count;
            return corners >= 3u ? uint3(0u, 1u, corners) : uint3(0u);
        }
        case MeshTopologyOp::DissolveVertices:
        case MeshTopologyOp::DissolveEdges:
        case MeshTopologyOp::DissolveFaces:
        case MeshTopologyOp::DissolveLimited: {
            if (TopoDissolveOwnLoop(ctx, job, f)) {
                const uint length = TopoMappedLoopLength(ctx, job, f);
                return length >= 3u ? uint3(0u, 1u, length) : uint3(0u);
            }
            const uint length = ctx.FaceLabels(job)[f] == f ? ctx.WalkLength(job)[f] : 0u;
            return length >= 3u ? uint3(0u, 1u, length) : uint3(0u);
        }
        default: {
            // A merge's targets are final only after the marks, so a loop under three corners drops here.
            if (ctx.FlagFaces(job)[f] == 0u) return uint3(0u);
            const uint length = TopoMappedLoopLength(ctx, job, f);
            return length >= 3u ? uint3(0u, 1u, length) : uint3(0u);
        }
    }
}

// Classify the explicit sparse selection once so topology rules only read compact flags.
kernel void TopologySelection(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshTopologyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const TopoContext ctx{bindless,pc};
    const uint2 tile=ctx.Tile(group_id);
    const MeshTopologyJob job=ctx.Jobs()[tile.x];
    const uint i=tile.y*ScanTileSize+lane;
    device uint *selected=ctx.Scratch()+job.SelectionOffset;
    const auto contains=[&](ElementWork work, uint h) { return WorkRank(bindless,work,h)!=InvalidOffset; };
    if (pc.PassParameter==0u) {
        if (i<job.SrcVertexCount) selected[i]=contains(job.VertexSelection,ctx.SrcVertexDomain(job).Handle(i));
    } else if (pc.PassParameter==1u) {
        const auto h=ctx.SrcHalfedgeDomain(job).Handle(i);
        if (h==InvalidOffset || !ctx.SrcEdgeFirst(job,h)) return;
        const auto e=ctx.SrcEdge(job,h);
        if (job.SelectionElement!=Element::Vertex) selected[job.SrcVertexCount+e]=contains(job.EdgeSelection,ctx.SrcEdgeDomain(job).Handle(e));
        else if (job.SelectionElement==Element::Vertex) {
            const auto corners=ctx.SrcCorners(job);
            selected[job.SrcVertexCount+e]=selected[corners[h]] && selected[corners[ctx.SrcPrev(job,h)]];
        }
    } else if (i<job.SrcFaceCount) {
        uint value=0u;
        if (job.SelectionElement!=Element::Vertex) value=contains(job.FaceSelection,ctx.SrcFaceDomain(job).Handle(i));
        else if (job.SelectionElement==Element::Vertex) {
            value=1u;
            const auto range=ctx.SrcFaceRange(job,i); const auto corners=ctx.SrcCorners(job);
            for (uint h=range.x;h<range.y;++h) value &= selected[corners[h]];
        }
        selected[job.SrcVertexCount+job.SrcHalfedgeCount+i]=value;
    }
}

kernel void TopologyZero(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshTopologyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const TopoContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshTopologyJob job = ctx.Jobs()[tile.x];
    const uint v = tile.y * ScanTileSize + lane;
    if (v >= job.SrcVertexCount) return;
    uint flags = job.Op == MeshTopologyOp::DissolveVertices && !(job.Flags & TopologyFlagListSelects) && ctx.SrcSelectedVertex(job, v) ? TopoDissolvable : 0u;
    if (job.Op==MeshTopologyOp::DeleteEdges && job.SelectionElement==Element::Vertex && ctx.SrcSelectedVertex(job,v)) flags |= TopoTagged;
    if (job.Op==MeshTopologyOp::DeleteLoose) flags |= TopoTagged;
    if ((TopologyCopiesSelection(job.Op) || job.Op==MeshTopologyOp::ExtrudeRegion) && ctx.SrcSelectedVertex(job,v)) flags |= TopoInRegion;
    ctx.FlagVertices(job)[v] = flags;
    ctx.VertexTargets(job)[v] = job.Op == MeshTopologyOp::MergeAtTarget && ctx.SrcSelectedVertex(job, v) ? job.TargetVertex : v;
    if (job.Op==MeshTopologyOp::Decimate) *ctx.Inward(job,v,0u)=packed_float3(ctx.SrcPosition(job,v));
    if (TopologyIsDissolve(job.Op)) {
        // Count canonical edges once, including users outside the emitted core.
        const auto src=ctx.Src(job);
        uint total=0u;
        for (const auto item:src.Fan(ctx.SrcVertexDomain(job).Handle(v))) {
            if (item.y==InvalidOffset) ++total;
            else {
                flags|=TopoSurfaceVertex;
                total+=uint(src.EdgeFirst(item.x))+uint(src.EdgeFirst(src.Next(item.x,item.y)));
            }
        }
        ctx.FlagVertices(job)[v]=flags;
        ctx.VertexEdgeTotal(job)[v] = total;
        ctx.VertexEdgeDissolved(job)[v] = 0u;
    }
    if (v == 0u) ctx.State(job)[0] = 0u;
}

// Flags the halfedge's edge when the dissolve removes it, counts its ends' edges, and marks the ends the dissolve may drop.
inline void TopoMarkDissolvedEdge(TopoContext ctx, MeshTopologyJob job, uint h) {
    const auto corners = ctx.SrcCorners(job);
    const uint to = corners[h], from = corners[ctx.SrcPrev(job, h)];
    const uint opposite = ctx.SrcOpposite(job, h);
    const bool wire=ctx.Src(job).HalfedgeFace(h)==InvalidOffset;
    const bool wire_removed=wire && job.Op==MeshTopologyOp::DissolveVertices &&
        ((ctx.SrcSelectedVertex(job,to) && (ctx.FlagVertices(job)[to]&TopoSurfaceVertex)) ||
         (ctx.SrcSelectedVertex(job,from) && (ctx.FlagVertices(job)[from]&TopoSurfaceVertex)));
    bool dissolved = !wire && opposite != InvalidOffset;
    if (dissolved) {
        switch (job.Op) {
            case MeshTopologyOp::DissolveVertices: dissolved = ctx.SrcSelectedVertex(job, to) || ctx.SrcSelectedVertex(job, from); break;
            case MeshTopologyOp::DissolveEdges:
            case MeshTopologyOp::RotateEdges: dissolved = ctx.SrcSelectedEdge(job, ctx.SrcEdge(job, h)); break;
            case MeshTopologyOp::DissolveLimited: {
                const uint f = ctx.SrcFaceOf(job, h), g = ctx.SrcFaceOf(job, opposite);
                dissolved = job.Param0>0.f && ctx.SrcSelectedEdge(job,ctx.SrcEdge(job,h)) &&
                    (!(job.Flags&TopologyFlagFaceSelection) || (ctx.SrcSelectedFace(job,f) && ctx.SrcSelectedFace(job,g))) &&
                    !TopoLimitedDelimiter(ctx,job,h,opposite) &&
                    dot(normalize(float3(ctx.SrcFaceNormals(job)[f])),normalize(float3(ctx.SrcFaceNormals(job)[g])))>cos(job.Param0);
                break;
            }
            default: dissolved = ctx.SrcSelectedFace(job, ctx.SrcFaceOf(job, h)) && ctx.SrcSelectedFace(job, ctx.SrcFaceOf(job, opposite)); break;
        }
    }
    if (dissolved) ctx.FlagHalfedges(job)[h] |= TopoDissolved;
    if (wire_removed) ctx.FlagHalfedges(job)[h] |= TopoWireRemoved;
    if (!ctx.SrcEdgeFirst(job, h)) return;
    if (!dissolved && !wire_removed) return;
    device atomic_uint *removed = ctx.Atomic(ctx.VertexEdgeDissolved(job));
    atomic_fetch_add_explicit(&removed[to], 1u, memory_order_relaxed);
    atomic_fetch_add_explicit(&removed[from], 1u, memory_order_relaxed);
    if (!dissolved || job.Op != MeshTopologyOp::DissolveEdges || (job.Flags & TopologyFlagKeepVertices)) return;
    device atomic_uint *flags = ctx.Atomic(ctx.FlagVertices(job));
    atomic_fetch_or_explicit(&flags[to], TopoDissolvable, memory_order_relaxed);
    atomic_fetch_or_explicit(&flags[from], TopoDissolvable, memory_order_relaxed);
}

// Collapses an edge shorter than the distance toward its lower vertex.
inline void TopoMarkShortEdge(TopoContext ctx, MeshTopologyJob job, uint h) {
    if (!ctx.SrcEdgeFirst(job, h)) return;
    const auto corners = ctx.SrcCorners(job);
    const uint a = corners[ctx.SrcPrev(job, h)], b = corners[h];
    if (!ctx.SrcSelectedVertex(job, a) || !ctx.SrcSelectedVertex(job, b)) return;
    if (distance(ctx.SrcPosition(job, a), ctx.SrcPosition(job, b)) >= job.Param0) return;
    device atomic_uint *targets = ctx.Atomic(ctx.VertexTargets(job));
    atomic_fetch_min_explicit(&targets[max(a, b)], min(a, b), memory_order_relaxed);
}

// Classifies each halfedge for its operator: deletion tags, the side quads and vertex copies an extrusion needs,
// the edges a dissolve or collapse removes, an edge split's sectors, and the cut parameters a listed subdivide fills.
kernel void TopologyMarkHalfedges(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshTopologyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const TopoContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshTopologyJob job = ctx.Jobs()[tile.x];
    const uint hi = tile.y * ScanTileSize + lane;
    const uint h = ctx.SrcHalfedgeDomain(job).Handle(hi);
    if (h == InvalidOffset) return;
    if ((TopologyIsMerge(job.Op) || TopologyIsDissolve(job.Op)) && hi<job.SrcEdgeCount) ctx.WireEdgeMap(job)[hi]=0u;
    if (job.Op==MeshTopologyOp::ExtrudeRegion && hi<job.SrcEdgeCount) {
        ctx.EdgeParams(job)[2u*hi]=0u;
        ctx.EdgeParams(job)[2u*hi+1u]=InvalidOffset;
    } else if (TopologyClassifiesEdges(job.Op) && hi<job.SrcEdgeCount)
        ctx.EdgeParams(job)[hi]=TopoEdgeDeleted(ctx,job,ctx.SrcEdgeHalfedge(job,hi)) ? 0u : InvalidOffset;
    device atomic_uint *flags = ctx.Atomic(ctx.FlagVertices(job));
    const auto corners = ctx.SrcCorners(job);
    const uint to = corners[h], from = corners[ctx.SrcPrev(job, h)];
    const auto mark_ends = [&](uint bits) {
        atomic_fetch_or_explicit(&flags[to], bits, memory_order_relaxed);
        atomic_fetch_or_explicit(&flags[from], bits, memory_order_relaxed);
    };
    uint halfedge_flags = 0u;
    switch (TopologyBaseOp(job.Op)) {
        case MeshTopologyOp::KeepSelectedFaces:
            if (ctx.Src(job).HalfedgeFace(h)==InvalidOffset && ctx.SrcSelectedEdge(job,ctx.SrcEdge(job,h))) mark_ends(TopoKept);
            break;
        case MeshTopologyOp::SplitGeometry:
        case MeshTopologyOp::DuplicateGeometry:
            if (TopologyCopiesSelection(job.Op) && ctx.SrcSelectedEdge(job,ctx.SrcEdge(job,h))) mark_ends(TopoInRegion);
            break;
        case MeshTopologyOp::DeleteEdges:
            if (ctx.SrcEdgeFirst(job, h) && ctx.SrcSelectedEdge(job, ctx.SrcEdge(job, h))) mark_ends(TopoTagged);
            break;
        case MeshTopologyOp::ExtrudeRegion: {
            if (job.Op==MeshTopologyOp::ExtrudeRegion) {
                if (ctx.SrcSelectedEdge(job,ctx.SrcEdge(job,h))) mark_ends(TopoInRegion|TopoRegionEdge);
                break;
            }
            if (!ctx.SrcSelectedFace(job, ctx.SrcFaceOf(job, h))) break;
            const uint opposite = ctx.SrcOpposite(job, h);
            const bool neighbor_selected = opposite != InvalidOffset && ctx.SrcSelectedFace(job, ctx.SrcFaceOf(job, opposite));
            if (neighbor_selected) break;
            halfedge_flags = TopoSide;
            if (opposite == InvalidOffset) {
                mark_ends(TopoOnBoundary);
            } else {
                mark_ends(TopoNeedsCopy);
                atomic_fetch_or_explicit(ctx.Atomic(ctx.State(job)), TopoDelOrig, memory_order_relaxed);
            }
            break;
        }
        case MeshTopologyOp::ExtrudeEdges:
            if (ctx.SrcEdgeFirst(job, h) && ctx.SrcSelectedEdge(job, ctx.SrcEdge(job, h))) {
                halfedge_flags = TopoSide | TopoSideFlip;
                mark_ends(TopoNeedsCopy);
            }
            break;
        default:
            break;
    }
    ctx.FlagHalfedges(job)[h] = halfedge_flags;
    if (TopologyIsDissolve(job.Op) && TopoLineJoins(job) && job.SrcFaceCount) ctx.DissolvePrevious(job)[h]=InvalidOffset;
    if (job.Op == MeshTopologyOp::Wireframe && hi < job.SrcEdgeCount) {
        TopoWireEdges(ctx, job)[2u * hi] = 0u;
        TopoWireEdges(ctx, job)[2u * hi + 1u] = InvalidOffset;
    }
    if (TopologyIsDissolve(job.Op)) TopoMarkDissolvedEdge(ctx, job, h);
    else if (job.Op == MeshTopologyOp::DissolveDegenerate) TopoMarkShortEdge(ctx, job, h);
    else if (job.Op == MeshTopologyOp::EdgeSplit) {
        if (ctx.SrcEdgeFirst(job,h) && ctx.SrcSelectedEdge(job,ctx.SrcEdge(job,h))) mark_ends(TopoTagged);
        ctx.HalfedgeAux(job)[h] = TopoSectorRep(ctx, job, h);
    }
    else if (job.Op == MeshTopologyOp::Subdivide && (job.Flags & (TopologyFlagListCuts | TopologyFlagScreenCuts)) && hi < job.SrcEdgeCount) ctx.EdgeParams(job)[hi] = InvalidOffset;
    else if (job.Op == MeshTopologyOp::Subdivide && (job.Flags & TopologyFlagListSelects) && hi < job.SrcEdgeCount) ctx.EdgeParams(job)[hi] = 0u;
}

// Flags each surviving source face and marks its corner vertices as kept or tagged.
kernel void TopologyMarkFaces(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshTopologyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const TopoContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshTopologyJob job = ctx.Jobs()[tile.x];
    const uint f = tile.y * ScanTileSize + lane;
    if (f >= job.SrcFaceCount) return;
    if (pc.PassParameter!=0u) {
        if (pc.PassParameter==4u) {
            if (TopologyIsDissolve(job.Op) && TopoLineJoins(job)) TopoDissolveBoundary(ctx,job,f);
        } else if (TopologyIsMerge(job.Op)) {
            if (pc.PassParameter==1u) { TopoPlanWeld(ctx,job,f); TopoWeldKeys(ctx,job,f); }
            else if (pc.PassParameter==2u) TopoWeldInsert(ctx,job,f);
            else TopoWeldDeduplicate(ctx,job,f);
        }
        return;
    }
    const bool deleted = TopoFaceDeleted(ctx, job, f);
    ctx.FlagFaces(job)[f] = deleted ? 0u : 1u;
    if (TopologyIsDissolve(job.Op)) {
        ctx.FaceLabels(job)[f] = f;
        ctx.RegionBoundary(job)[f] = 0u;
        ctx.RegionStart(job)[f] = InvalidOffset;
        ctx.WalkLength(job)[f] = 0u;
    }
    if (job.Op == MeshTopologyOp::TrisToQuads) {
        ctx.FaceLabels(job)[f] = InvalidOffset;
        ctx.WalkLength(job)[f] = InvalidOffset;
    }
    const uint2 range = ctx.SrcFaceRange(job, f);
    device atomic_uint *flags = ctx.Atomic(ctx.FlagVertices(job));
    const auto corners = ctx.SrcCorners(job);
    const uint mark = (deleted ? TopoTagged : TopoKept) | (ctx.SrcSelectedFace(job, f) ? TopoInRegion : 0u);
    for (uint h = range.x; h < range.y; ++h) atomic_fetch_or_explicit(&flags[corners[h]], mark, memory_order_relaxed);
    if (job.Op == MeshTopologyOp::Wireframe && ctx.SrcSelectedFace(job, f)) {
        for (uint h = range.x; h < range.y; ++h) {
            device atomic_uint *edge = ctx.Atomic(TopoWireEdges(ctx, job)) + 2u * ctx.SrcEdge(job, h);
            atomic_fetch_add_explicit(edge, 1u, memory_order_relaxed);
            atomic_store_explicit(edge + 1u, h, memory_order_relaxed);
        }
    }
    // A bevel removes every vertex it moves corners off, marked as in the region.
    if (TopologyIsBevel(job.Op)) {
        for (uint h = range.x; h < range.y; ++h) {
            if (TopoVertexBeveled(ctx, job, corners[h])) atomic_fetch_or_explicit(&flags[corners[h]], TopoInRegion, memory_order_relaxed);
        }
    }
}

// Each affected vertex visits its complete fan once, including faces outside the edit core.
// Any surviving face keeps both incident edges as surface edges, including nonmanifold users.
kernel void TopologyMarkRetainedEdges(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshTopologyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const TopoContext ctx{bindless,pc};
    const uint2 tile=ctx.Tile(group_id);
    const MeshTopologyJob job=ctx.Jobs()[tile.x];
    const uint v=tile.y*ScanTileSize+lane;
    if (!TopologyClassifiesEdges(job.Op) || v>=job.SrcVertexCount) return;
    const auto src=ctx.Src(job);
    bool kept=false;
    for (const auto item:src.Fan(ctx.SrcVertexDomain(job).Handle(v))) {
        const uint f=ctx.SrcFaceDomain(job).Index(item.y);
        const bool surface=item.y!=InvalidOffset && (f==InvalidOffset || ctx.FlagFaces(job)[f]);
        const uint2 corners={item.x,item.y==InvalidOffset ? item.x : src.Next(item.x,item.y)};
        for (uint k=0u;k<(item.y==InvalidOffset ? 1u : 2u);++k) {
            const auto e=ctx.SrcEdge(job,corners[k]);
            if (job.Op==MeshTopologyOp::ExtrudeRegion) {
                // One endpoint counts each face user, including users outside the edit core.
                if (e==InvalidOffset || item.y==InvalidOffset ||
                    v!=min(ctx.SrcCorners(job)[corners[k]],ctx.SrcCorners(job)[src.Previous(corners[k])])) continue;
                if (ctx.SrcSelectedFace(job,f)) {
                    atomic_fetch_add_explicit(ctx.Atomic(ctx.EdgeParams(job))+2u*e,1u,memory_order_relaxed);
                    atomic_fetch_min_explicit(ctx.Atomic(ctx.EdgeParams(job))+2u*e+1u,corners[k],memory_order_relaxed);
                } else {
                    atomic_fetch_or_explicit(ctx.Atomic(ctx.EdgeParams(job))+2u*e,TopoRegionUnselected,memory_order_relaxed);
                    if (ctx.SrcSelectedEdge(job,e)) atomic_fetch_or_explicit(ctx.Atomic(ctx.State(job)),TopoDelOrig,memory_order_relaxed);
                }
                continue;
            }
            if (TopologyCopiesSelection(job.Op)) {
                const bool selected_face=ctx.SrcSelectedFace(job,f);
                if (e!=InvalidOffset && selected_face) atomic_store_explicit(ctx.Atomic(ctx.EdgeParams(job))+e,0u,memory_order_relaxed);
                kept |= item.y!=InvalidOffset ? !selected_face : !ctx.SrcSelectedEdge(job,e);
                continue;
            }
            if (TopoEdgeDeleted(ctx,job,corners[k])) continue;
            kept |= job.Op!=MeshTopologyOp::DeleteFaces || item.y==InvalidOffset || surface;
            if (surface && e!=InvalidOffset) atomic_store_explicit(ctx.Atomic(ctx.EdgeParams(job))+e,0u,memory_order_relaxed);
        }
    }
    if (kept) {
        const uint flags=job.Op==MeshTopologyOp::SplitGeometry && (ctx.FlagVertices(job)[v]&TopoInRegion) ? TopoNeedsCopy : TopoKept;
        atomic_fetch_or_explicit(ctx.Atomic(ctx.FlagVertices(job))+v,flags,memory_order_relaxed);
    }
}

// Pulls both ends of each linking edge to the lower of their labels: a dissolved edge's two face labels, or a collapsing edge's two vertex targets.
kernel void TopologyLink(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshTopologyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const TopoContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshTopologyJob job = ctx.Jobs()[tile.x];
    const uint hi = tile.y * ScanTileSize + lane;
    const uint h = ctx.SrcHalfedgeDomain(job).Handle(hi);
    const bool dissolve = TopologyIsDissolve(job.Op);
    if (h == InvalidOffset || !(dissolve || job.Op == MeshTopologyOp::MergeCollapse)) return;
    if (dissolve ? !TopoEdgeDissolved(ctx, job, h) : !ctx.SrcEdgeFirst(job, h)) return;
    const auto corners = ctx.SrcCorners(job);
    const uint a = dissolve ? ctx.SrcFaceOf(job, h) : corners[ctx.SrcPrev(job, h)];
    const uint b = dissolve ? ctx.SrcFaceOf(job, ctx.SrcOpposite(job, h)) : corners[h];
    if (!dissolve && (!ctx.SrcSelectedVertex(job, a) || !ctx.SrcSelectedVertex(job, b))) return;
    device uint *labels = dissolve ? ctx.FaceLabels(job) : ctx.VertexTargets(job);
    device atomic_uint *atomic_labels = ctx.Atomic(labels);
    const uint m = min(labels[a], labels[b]);
    const bool changed = atomic_fetch_min_explicit(&atomic_labels[a], m, memory_order_relaxed) > m || atomic_fetch_min_explicit(&atomic_labels[b], m, memory_order_relaxed) > m;
    if (changed) ctx.State(job)[1] = 1u;
}

// Jumps each label to its label's label: the pass parameter selects a dissolve's face labels or a merge's vertex targets.
kernel void TopologyJump(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshTopologyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const TopoContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshTopologyJob job = ctx.Jobs()[tile.x];
    const uint i = tile.y * ScanTileSize + lane;
    if (pc.PassParameter==2u) {
        if (job.Op==MeshTopologyOp::DissolveLimited && i<job.SrcVertexCount) TopoLimitedJump(ctx,job,i);
        return;
    }
    const bool faces = pc.PassParameter == 0u;
    const bool merge = job.Op == MeshTopologyOp::MergeByDistance || job.Op == MeshTopologyOp::MergeCollapse || job.Op == MeshTopologyOp::DissolveDegenerate;
    if (faces ? !TopologyIsDissolve(job.Op) || i >= job.SrcFaceCount : !merge || i >= job.SrcVertexCount) return;
    device uint *labels = faces ? ctx.FaceLabels(job) : ctx.VertexTargets(job);
    const uint label = labels[i], root = labels[label];
    if (root == label) return;
    labels[i] = root;
    ctx.State(job)[1] = 1u;
}

// Ends the label rounds once one changes nothing: clears every job's changed flag and, when none was set, every domain's indirect dispatch arguments.
// The pass parameter packs the job count above the domain count.
kernel void TopologyConverge(
    uint lane [[thread_index_in_threadgroup]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshTopologyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    if (lane != 0u) return;
    const TopoContext ctx{bindless, pc};
    const uint jobs = pc.PassParameter >> 8u, domains = pc.PassParameter & 255u;
    bool changed = false;
    for (uint j = 0u; j < jobs; ++j) {
        device uint *state = ctx.State(ctx.Jobs()[j]);
        changed = changed || state[1] != 0u;
        state[1] = 0u;
    }
    if (changed) return;
    for (uint j=0u;j<jobs;++j) if (ctx.Jobs()[j].Op==MeshTopologyOp::DissolveLimited)
        ctx.State(ctx.Jobs()[j])[0]|=TopoLabelsConverged;
    device uint *arguments = ctx.Scratch();
    for (uint w = 0u; w < 3u * domains; ++w) arguments[w] = 0u;
}

// Sums each region's boundary halfedges and finds its lowest one.
kernel void TopologyDissolveRegions(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshTopologyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const TopoContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshTopologyJob job = ctx.Jobs()[tile.x];
    const uint f = tile.y * ScanTileSize + lane;
    if (!TopologyIsDissolve(job.Op) || f >= job.SrcFaceCount) return;
    const uint root = ctx.FaceLabels(job)[f];
    const uint2 range = ctx.SrcFaceRange(job, f);
    uint boundary = 0u, first = InvalidOffset;
    for (uint h = range.x; h < range.y; ++h) {
        if (TopoEdgeDissolved(ctx, job, h)) continue;
        ++boundary;
        first = min(first, h);
    }
    if (boundary == 0u) return;
    atomic_fetch_add_explicit(&ctx.Atomic(ctx.RegionBoundary(job))[root], boundary, memory_order_relaxed);
    atomic_fetch_min_explicit(&ctx.Atomic(ctx.RegionStart(job))[root], first, memory_order_relaxed);
}

// Walks each region root's boundary to count the corners its merged face keeps.
kernel void TopologyDissolveWalk(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshTopologyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const TopoContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshTopologyJob job = ctx.Jobs()[tile.x];
    const uint f = tile.y * ScanTileSize + lane;
    if (!TopologyIsDissolve(job.Op) || f >= job.SrcFaceCount || ctx.FaceLabels(job)[f] != f) return;
    ctx.WalkLength(job)[f] = TopoWalkRegion(ctx, job, f, InvalidOffset, InvalidOffset);
}

// Restores the edges of every region whose boundary walk failed, so its faces come through unchanged.
kernel void TopologyDissolveRevert(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshTopologyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const TopoContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshTopologyJob job = ctx.Jobs()[tile.x];
    const uint hi = tile.y * ScanTileSize + lane;
    const uint h = ctx.SrcHalfedgeDomain(job).Handle(hi);
    if (!TopologyIsDissolve(job.Op) || h == InvalidOffset || !TopoEdgeDissolved(ctx, job, h)) return;
    if (ctx.WalkLength(job)[ctx.FaceLabels(job)[ctx.SrcFaceOf(job, h)]] != 0u) return;
    ctx.FlagHalfedges(job)[h] &= ~TopoDissolved;
    if (!ctx.SrcEdgeFirst(job, h)) return;
    const auto corners = ctx.SrcCorners(job);
    device atomic_uint *removed = ctx.Atomic(ctx.VertexEdgeDissolved(job));
    atomic_fetch_sub_explicit(&removed[corners[h]], 1u, memory_order_relaxed);
    atomic_fetch_sub_explicit(&removed[corners[ctx.SrcPrev(job, h)]], 1u, memory_order_relaxed);
}

// Empties a job's table, whose entries alone the host tiles.
kernel void TopologyMergeTable(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshTopologyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const TopoContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshTopologyJob job = ctx.Jobs()[tile.x];
    const uint i = tile.y * ScanTileSize + lane;
    if (i > job.TableMask) return;
    ctx.Table(job)[i] = InvalidOffset;
}

// Inserts each selected vertex into the table at its grid cell's probe run.
kernel void TopologyMergeInsert(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshTopologyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const TopoContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshTopologyJob job = ctx.Jobs()[tile.x];
    const uint v = tile.y * ScanTileSize + lane;
    if (job.Op != MeshTopologyOp::MergeByDistance || v >= job.SrcVertexCount || !ctx.SrcSelectedVertex(job, v)) return;
    device atomic_uint *table = ctx.Atomic(ctx.Table(job));
    uint slot = TopoCellHash(TopoMergeCell(ctx.SrcPosition(job, v), job.Param0)) & job.TableMask;
    for (uint probe = 0u; probe <= job.TableMask; ++probe) {
        uint empty = InvalidOffset;
        if (atomic_compare_exchange_weak_explicit(&table[slot], &empty, v, memory_order_relaxed, memory_order_relaxed)) return;
        slot = (slot + 1u) & job.TableMask;
    }
}

// Each selected vertex targets the lowest selected vertex within the merge distance, found among its neighboring cells.
kernel void TopologyMergeQuery(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshTopologyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const TopoContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshTopologyJob job = ctx.Jobs()[tile.x];
    const uint v = tile.y * ScanTileSize + lane;
    if (job.Op != MeshTopologyOp::MergeByDistance || v >= job.SrcVertexCount || !ctx.SrcSelectedVertex(job, v)) return;
    const float3 p = ctx.SrcPosition(job, v);
    const float distance_sq = job.Param0 * job.Param0;
    const int3 cell = TopoMergeCell(p, job.Param0);
    device const uint *table = ctx.Table(job);
    uint target = v;
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const int3 neighbor = cell + int3(dx, dy, dz);
                uint slot = TopoCellHash(neighbor) & job.TableMask;
                for (uint probe = 0u; probe <= job.TableMask; ++probe) {
                    const uint other = table[slot];
                    if (other == InvalidOffset) break;
                    if (other < target && all(TopoMergeCell(ctx.SrcPosition(job, other), job.Param0) == neighbor) && distance_squared(ctx.SrcPosition(job, other), p) <= distance_sq) target = other;
                    slot = (slot + 1u) & job.TableMask;
                }
            }
        }
    }
    ctx.VertexTargets(job)[v] = target;
}

// Records mapped edges, preferring a surviving face and otherwise the lowest loose-edge representative.
kernel void TopologyLineKeys(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshTopologyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const TopoContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshTopologyJob job = ctx.Jobs()[tile.x];
    const uint hi = tile.y * ScanTileSize + lane;
    const uint h = ctx.SrcHalfedgeDomain(job).Handle(hi);
    if (h == InvalidOffset || !TopoLineJoins(job)) return;
    const bool face=ctx.Src(job).HalfedgeFace(h)!=InvalidOffset;
    if (!TopoLineSurface(ctx,job,h) && !(TopologyIsDissolve(job.Op) && face) && !ctx.SrcEdgeFirst(job,h)) return;
    const TopoLine line = TopoLineOutput(ctx, job, h);
    if (!line.Kept) return;
    const uint priority=TopoLinePriority(ctx,job,h,line);
    const uint2 key = TopoLineKey(line);
    device atomic_uint *table = ctx.Atomic(ctx.Table(job));
    uint slot = TopoLineHash(key) & job.TableMask;
    for (uint probe = 0u; probe <= job.TableMask;) {
        uint occupant = InvalidOffset;
        if (atomic_compare_exchange_weak_explicit(&table[slot], &occupant, hi, memory_order_relaxed, memory_order_relaxed)) return;
        if (occupant == InvalidOffset) continue; // Weak CAS may fail spuriously.
        if (all(TopoLineKey(TopoLineOutput(ctx, job, ctx.SrcHalfedgeDomain(job).Handle(occupant))) == key)) {
            const uint other=ctx.SrcHalfedgeDomain(job).Handle(occupant);
            const uint other_priority=TopoLinePriority(ctx,job,other,TopoLineOutput(ctx,job,other));
            if (other_priority<priority || (priority==other_priority && occupant<=hi)) return;
            if (atomic_compare_exchange_weak_explicit(&table[slot],&occupant,hi,memory_order_relaxed,memory_order_relaxed)) return;
            continue;
        }
        slot = (slot + 1u) & job.TableMask;
        ++probe;
    }
}

// A limited dissolve drops vertices left with two edges that nearly continue each other.
kernel void TopologyFinalizeVertices(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshTopologyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const TopoContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshTopologyJob job = ctx.Jobs()[tile.x];
    const uint v = tile.y * ScanTileSize + lane;
    if (v>=job.SrcVertexCount) return;
    if (pc.PassParameter==1u) {
        if (job.Op==MeshTopologyOp::DissolveLimited) TopoLimitedSimplify(ctx,job,v);
        return;
    }
    if (job.Op==MeshTopologyOp::ExtrudeRegion) {
        uint flags=ctx.FlagVertices(job)[v];
        if (!(flags&TopoInRegion)) return;
        const auto src=ctx.Src(job);
        for (const auto item:src.Fan(ctx.SrcVertexDomain(job).Handle(v))) {
            const uint2 corners={item.x,item.y==InvalidOffset ? item.x : src.Next(item.x,item.y)};
            for (uint k=0u;k<(item.y==InvalidOffset ? 1u : 2u);++k) {
                const uint e=ctx.SrcEdge(job,corners[k]);
                if (!ctx.SrcSelectedEdge(job,e)) { flags|=TopoNeedsCopy; continue; }
                if (ctx.EdgeParams(job)[2u*e]&TopoRegionUnselected) flags|=TopoNeedsCopy;
                if (TopoRegionEdgeFaces(ctx,job,e)<2u) {
                    flags|=TopoOnBoundary|TopoNeedsCopy;
                    atomic_fetch_or_explicit(ctx.Atomic(ctx.State(job)),TopoRegionSides,memory_order_relaxed);
                }
            }
        }
        if (!(flags&TopoRegionEdge)) flags|=TopoOnBoundary|TopoNeedsCopy;
        ctx.FlagVertices(job)[v]=flags;
        return;
    }
    if (job.Op==MeshTopologyOp::DissolveLimited) TopoLimitedPrepare(ctx,job,v);
}

// Fills a job's list into scratch: a selection list flags its vertices or a subdivide's edges, a cut list sets its edge parameters,
// and a knife sets the parameter of every edge whose screen segment crosses it.
kernel void TopologyListFill(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshTopologyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const TopoContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshTopologyJob job = ctx.Jobs()[tile.x];
    if (pc.PassParameter==1u && job.Op!=MeshTopologyOp::DissolveVertices) return;
    const uint i = tile.y * ScanTileSize + lane;
    if (job.Flags & TopologyFlagScreenCuts) {
        if (i >= job.SrcEdgeCount) return;
        const auto corners = ctx.SrcCorners(job);
        const uint h = ctx.SrcEdgeHalfedge(job, i);
        const float4x4 to_clip = job.ScreenTransform.Unpack();
        const float4 ca = to_clip * float4(ctx.SrcPosition(job, corners[ctx.SrcPrev(job, h)]), 1.f), cb = to_clip * float4(ctx.SrcPosition(job, corners[h]), 1.f);
        if (ca.w <= 0.f || cb.w <= 0.f) return;
        const float2 extent = float2(job.Extent);
        const float2 a = float2(ca.x / ca.w + 1.f, 1.f - ca.y / ca.w) * 0.5f * extent, b = float2(cb.x / cb.w + 1.f, 1.f - cb.y / cb.w) * 0.5f * extent;
        // Segment intersection in the plane: the edge's parameter is the cut's parameter along its representative halfedge.
        const float2 d = b - a, k = float2(job.KnifeEnd) - float2(job.KnifeStart), w = float2(job.KnifeStart) - a;
        const float denominator = d.x * k.y - d.y * k.x;
        if (abs(denominator) < 1e-12f) return;
        const float t = (w.x * k.y - w.y * k.x) / denominator, u = (w.x * d.y - w.y * d.x) / denominator;
        if (t <= 0.f || t >= 1.f || u < 0.f || u > 1.f) return;
        ctx.EdgeParams(job)[i] = as_type<uint>(t);
        return;
    }
    if (job.ListOffset == InvalidOffset) return;
    device const uint *list = ctx.Lists(job);
    if (i >= list[0]) return;
    if (job.Op==MeshTopologyOp::Decimate) {
        const uint v=list[1u+5u*i],target=list[2u+5u*i];
        ctx.VertexTargets(job)[v]=target;
        ctx.FlagVertices(job)[v]|=TopoListed;
        if (v==target) *ctx.Inward(job,v,0u)=packed_float3(as_type<float>(list[3u+5u*i]),as_type<float>(list[4u+5u*i]),as_type<float>(list[5u+5u*i]));
        return;
    }
    if ((job.Flags & TopologyFlagListSelects) || job.Op == MeshTopologyOp::ReplaceFaces) {
        const uint element = list[1u + i];
        if (job.Op == MeshTopologyOp::Subdivide) {
            if (element < job.SrcEdgeCount) ctx.EdgeParams(job)[element] = 1u;
        } else if (element < job.SrcVertexCount) {
            atomic_fetch_or_explicit(&ctx.Atomic(ctx.FlagVertices(job))[element], TopoListed | (job.Op == MeshTopologyOp::DissolveVertices ? TopoDissolvable : 0u), memory_order_relaxed);
        }
        return;
    }
    if (!(job.Flags & TopologyFlagListCuts) || job.Op != MeshTopologyOp::Subdivide) return;
    const uint edge = list[1u + 2u * i];
    if (edge < job.SrcEdgeCount) ctx.EdgeParams(job)[edge] = list[2u + 2u * i];
}

// Each unmatched selected triangle records its lowest-cost unmatched neighbor, by cost then by halfedge.
kernel void TopologyJoinBest(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshTopologyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const TopoContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshTopologyJob job = ctx.Jobs()[tile.x];
    const uint f = tile.y * ScanTileSize + lane;
    if (job.Op != MeshTopologyOp::TrisToQuads || f >= job.SrcFaceCount) return;
    device uint *best = ctx.WalkLength(job);
    best[f] = InvalidOffset;
    const uint2 range = ctx.SrcFaceRange(job, f);
    if (range.y - range.x != 3u || !ctx.SrcSelectedFace(job, f) || ctx.FaceLabels(job)[f] != InvalidOffset) return;
    float best_cost = 3.4e38f;
    for (uint h = range.x; h < range.y; ++h) {
        const uint opposite = ctx.SrcOpposite(job, h);
        const uint neighbor = ctx.SrcFaceOf(job, opposite);
        if (neighbor == InvalidOffset || ctx.FaceLabels(job)[neighbor] != InvalidOffset) continue;
        const float cost = TopoJoinCost(ctx, job, h);
        if (cost >= 0.f && cost < best_cost) {
            best_cost = cost;
            best[f] = h;
        }
    }
}

// Two triangles that recorded each other join, and their other neighbors record again next round.
kernel void TopologyJoinMatch(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshTopologyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const TopoContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshTopologyJob job = ctx.Jobs()[tile.x];
    const uint f = tile.y * ScanTileSize + lane;
    if (job.Op != MeshTopologyOp::TrisToQuads || f >= job.SrcFaceCount) return;
    const uint h = ctx.WalkLength(job)[f];
    if (h == InvalidOffset) return;
    const uint opposite = ctx.SrcOpposite(job, h);
    const uint g = ctx.SrcFaceOf(job, opposite);
    if (g == InvalidOffset || ctx.WalkLength(job)[g] != opposite) return;
    ctx.FaceLabels(job)[f] = g;
    ctx.RegionStart(job)[f] = h;
    ctx.State(job)[1] = 1u;
}

// Clears the inward vectors the gather adds to output vertex positions.
kernel void TopologyZeroVertices(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshTopologyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const TopoContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshTopologyJob job = ctx.Jobs()[tile.x];
    const uint d = tile.y * ScanTileSize + lane;
    if (!TopologyDisplaces(job.Op, job.Flags) || job.Op==MeshTopologyOp::Decimate || d >= job.DstVertexCount) return;
    *ctx.Inward(job, d, 0u) = packed_float3(0.f);
    *ctx.Inward(job, d, 1u) = packed_float3(0.f);
}

kernel void TopologyCountVertices(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshTopologyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const TopoContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshTopologyJob job = ctx.Jobs()[tile.x];
    const uint v = tile.y * ScanTileSize + lane;
    if (v >= job.SrcVertexCount) return;
    const uint entry = ctx.VertexEntry(v);
    uint own = job.Op == MeshTopologyOp::EdgeSplit ? TopoSectorCount(ctx, job, v) : (TopoVertexKept(ctx, job, v) ? 1u : 0u);
    if (job.Op == MeshTopologyOp::Wireframe && (job.Flags & TopologyFlagWireBoundary) && (ctx.FlagVertices(job)[v] & TopoInRegion)) {
        uint2 edges;
        if (TopoWireBoundaryEdges(ctx, job, v, edges)) ctx.FlagVertices(job)[v] |= TopoOnBoundary;
    }
    uint faces = 0u, corners = 0u;
    if (TopologyIsBevel(job.Op) && (ctx.FlagVertices(job)[v] & TopoInRegion)) {
        // The boundary indexes halfedge outputs. Interior profile rings belong
        // to this vertex entry and are counted by the same scan.
        corners = TopoBevelRing(ctx, job, v, [](uint2, uint, uint) {}).Length;
        if (corners >= 3u) {
            const uint rings = job.Op == MeshTopologyOp::BevelVertices ? TopoBevelSegments(job) - 1u : 0u;
            own = rings * corners;
            faces = 1u + rings * corners;
            corners *= 1u + 4u * rings;
        } else corners = 0u;
    }
    // A line extrusion joins each vertex to its copy.
    const uint copies = TopoVertexCopies(ctx, job, v);
    const uint wires=job.Op==MeshTopologyOp::ExtrudeRegion ? (TopoRegionConnectsVertex(ctx,job,v) ? 2u*job.Steps : 0u) :
        (TopoLineCore(job) && !TopologyCopiesSelection(job.Op) && !TopologyExtrudesSides(job.Op) ? 2u*copies : 0u);
    ctx.WriteCounts(job, entry, uint3(own + copies, faces, corners), wires);
}

kernel void TopologyCountHalfedges(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshTopologyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const TopoContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshTopologyJob job = ctx.Jobs()[tile.x];
    const uint hi = tile.y * ScanTileSize + lane;
    const uint h = ctx.SrcHalfedgeDomain(job).Handle(hi);
    if (h == InvalidOffset) return;
    const uint entry = ctx.HalfedgeEntry(job, h);
    const bool side = TopoHalfedgeMakesSide(ctx, job, h);
    if (TopologyCopiesSelection(job.Op)) {
        const uint e=ctx.SrcEdge(job,h);
        const bool own=ctx.Src(job).HalfedgeFace(h)==InvalidOffset && (job.Op!=MeshTopologyOp::SplitGeometry || !ctx.SrcSelectedEdge(job,e));
        const bool copy=ctx.SrcSelectedEdge(job,e) && ctx.EdgeParams(job)[e]==InvalidOffset;
        ctx.WriteCounts(job,entry,uint3(0),ctx.SrcEdgeFirst(job,h) ? 2u*(uint(own)+uint(copy)) : 0u);
        return;
    }
    if (TopologyIsDelete(job.Op)) {
        const bool wire=job.Op!=MeshTopologyOp::DeleteFaces && ctx.SrcEdgeFirst(job,h) && ctx.EdgeParams(job)[ctx.SrcEdge(job,h)]==InvalidOffset;
        ctx.WriteCounts(job,entry,uint3(0),wire ? 2u : 0u);
        return;
    }
    if (job.Op==MeshTopologyOp::AddPrimitives && ctx.Src(job).HalfedgeFace(h)==InvalidOffset) {
        ctx.WriteCounts(job,entry,uint3(0));
        return;
    }
    const bool cut = job.Op == MeshTopologyOp::Subdivide && ctx.SrcEdgeFirst(job, h) && TopoEdgeCut(ctx, job, h);
    if (TopologyExtrudesSides(job.Op) && ctx.Src(job).HalfedgeFace(h)==InvalidOffset && !side) {
        ctx.WriteCounts(job,entry,uint3(0u),ctx.SrcEdgeFirst(job,h) ? 2u : 0u);
        return;
    }
    if (TopologyIsMerge(job.Op) || (TopologyIsDissolve(job.Op) && (TopoLineJoins(job) || ctx.Src(job).HalfedgeFace(h)==InvalidOffset)) || (TopoLineCore(job) && !TopologyExtrudesSides(job.Op)) ||
        ((job.Op==MeshTopologyOp::Subdivide || job.Op==MeshTopologyOp::KeepSelectedFaces) && ctx.Src(job).HalfedgeFace(h)==InvalidOffset)) {
        ctx.WriteCounts(job, entry, uint3(cut ? TopoSubdivideCuts(job) : 0u, 0u, 0u), (ctx.SrcEdgeFirst(job,h) || (TopologyIsDissolve(job.Op) && ctx.Src(job).HalfedgeFace(h)!=InvalidOffset)) ? TopoLineCorners(ctx,job,h) : 0u);
        return;
    }
    if (TopologyIsBevel(job.Op)) {
        const BevelHalfedgeOutputs o = TopoBevelOutputs(ctx, job, h);
        const bool strip = ctx.SrcEdgeFirst(job, h) && TopoEdgeBeveled(ctx, job, h);
        const uint segments = strip ? TopoBevelSegments(job) : 0u;
        ctx.WriteCounts(job, entry, uint3(o.Count(), segments, 4u * segments));
        return;
    }
    ctx.WriteCounts(job, entry, uint3(cut ? TopoSubdivideCuts(job) : 0u, side ? job.Steps : 0u, side ? 4u * job.Steps : 0u));
}

// Counts each face's outputs and writes the scan terminator.
kernel void TopologyCountFaces(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshTopologyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const TopoContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshTopologyJob job = ctx.Jobs()[tile.x];
    const uint f = tile.y * ScanTileSize + lane;
    if (f > job.SrcFaceCount + 1u) return;
    const uint entry = ctx.FaceEntry(job, f);
    if (f < job.SrcFaceCount) {
        ctx.WriteCounts(job, entry, TopoFaceOutputs(ctx, job, f));
        return;
    }
    // The list entry carries the vertices and faces a job appends from its list, and the terminator entry holds zeros.
    uint3 listed = uint3(0u);
    uint wire_corners=0u;
    if (f == job.SrcFaceCount && job.Op == MeshTopologyOp::AddPrimitives && job.ListOffset != InvalidOffset) {
        device const uint *list = ctx.Lists(job);
        listed.x = list[0];
        uint cursor = TopologyPrimitiveHeaderWords + list[1];
        const uint primitives = list[cursor++];
        for (uint i = 0u; i < primitives; ++i) {
            const uint n=list[cursor++];
            if (n==2u) wire_corners+=2u;
            else { ++listed.y; listed.z+=n; }
            // An existing face may emit the representative of a newly selected boundary edge.
            for (uint k=0u;k<n;++k) {
                const uint h=list[cursor+2u*k+1u];
                if (h!=InvalidOffset) {
                    const auto src=ctx.Src(job);
                    ctx.FlagHalfedges(job)[src.EdgeHalfedge(src.Edge(h))] |= TopoListed;
                }
            }
            cursor += 2u*n;
        }
    }
    ctx.WriteCounts(job, entry, listed, wire_corners);
}

// Count arrays and their block sums, shared by the four allocation quantities.
struct TopoScan {
    device uint *Counts;
    uint Entries;
    device uint *Blocks;
    uint BlockCount;
};

inline TopoScan TopoScanOf(TopoContext ctx, MeshTopologyJob job, uint quantity) {
    return {ctx.Counts(job, quantity), job.CountEntries, ctx.Scratch() + job.CountBlockOffset + quantity * job.CountBlockCount, job.CountBlockCount};
}
kernel void TopologyScanBlockSum(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshTopologyPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    threadgroup uint *sums [[threadgroup(0)]]
) {
    const TopoContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshTopologyJob job = ctx.Jobs()[tile.x];
    const TopoScan probe = TopoScanOf(ctx, job, 0u);
    const uint quantity = tile.y / probe.BlockCount, block = tile.y % probe.BlockCount;
    const TopoScan scan = TopoScanOf(ctx, job, quantity);
    ScanBlockSum(scan.Counts, scan.Entries, block, scan.Blocks, lane, simd_lane, simd_group, sums);
}

kernel void TopologyScanBlockPrefix(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshTopologyPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    threadgroup uint *sums [[threadgroup(0)]]
) {
    const TopoContext ctx{bindless, pc};
    const MeshTopologyJob job = ctx.Jobs()[group_id];
    // The count scan runs its quantities quantity-major over the count blocks.
    const uint quantities = 4u;
    for (uint quantity = 0u; quantity < quantities; ++quantity) {
        const TopoScan scan = TopoScanOf(ctx, job, quantity);
        ScanBlockPrefix(scan.Blocks, scan.BlockCount, lane, simd_lane, simd_group, sums);
    }
}

// Each thread overwrites only its own counts with their exclusive offsets, permitting an in-place scan.
kernel void TopologyScanOffsets(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshTopologyPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    threadgroup uint *sums [[threadgroup(0)]]
) {
    const TopoContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshTopologyJob job = ctx.Jobs()[tile.x];
    const TopoScan probe = TopoScanOf(ctx, job, 0u);
    const uint quantity = tile.y / probe.BlockCount, block = tile.y % probe.BlockCount;
    const TopoScan scan = TopoScanOf(ctx, job, quantity);
    ScanBlockOffsets(scan.Counts, scan.Entries, block, scan.Blocks, scan.Counts, lane, simd_lane, simd_group, sums);
}

kernel void TopologyScatterVertices(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshTopologyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const TopoContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshTopologyJob job = ctx.Jobs()[tile.x];
    const uint v = tile.y * ScanTileSize + lane;
    if (v >= job.SrcVertexCount) return;
    const uint entry = ctx.VertexEntry(v);
    device const uint *offsets = ctx.Counts(job, TopoCountVertices);
    const uint count = offsets[entry + 1u] - offsets[entry];
    if (TopologyIsBevel(job.Op)) {
        // Every profile ring is built from the same counted boundary. The last
        // ring receives the cap, wound against the surrounding quads.
        device const uint *face_offsets = ctx.Counts(job, TopoCountFaces);
        if (face_offsets[entry + 1u] == face_offsets[entry]) return;
        const uint segments = job.Op == MeshTopologyOp::BevelVertices ? TopoBevelSegments(job) : 1u;
        const uint fd = face_offsets[entry], base = ctx.Counts(job, TopoCountCorners)[entry];
        const uint length = (ctx.Counts(job, TopoCountCorners)[entry + 1u] - base) /
            (1u + 4u * (segments - 1u));
        uint2 first_id = uint2(InvalidOffset), previous_id = uint2(InvalidOffset);
        uint first_source = InvalidOffset, previous_source = InvalidOffset;
        const auto quad = [&](uint2 current, uint current_source, uint current_index, uint2 previous, uint previous_source) {
            for (uint ring = 0u; ring + 1u < segments; ++ring) {
                const uint current_outer = TopoBevelVertexRingIndex(ctx, job, entry, current, current_index, length, ring);
                const uint previous_outer = TopoBevelVertexRingIndex(ctx, job, entry, previous, (current_index + length - 1u) % length, length, ring);
                const uint current_inner = TopoBevelVertexRingIndex(ctx, job, entry, current, current_index, length, ring + 1u);
                const uint previous_inner = TopoBevelVertexRingIndex(ctx, job, entry, previous, (current_index + length - 1u) % length, length, ring + 1u);
                const uint face = fd + ring * length + (current_index + length - 1u) % length;
                const uint corner = base + 4u * (ring * length + (current_index + length - 1u) % length);
                ctx.WriteCorner(job, corner, current_outer, current_source, current_source, 0.f, InvalidOffset, true);
                ctx.WriteCorner(job, corner + 1u, previous_outer, previous_source, previous_source, 0.f, InvalidOffset, true);
                ctx.WriteCorner(job, corner + 2u, previous_inner, previous_source, previous_source, 0.f, InvalidOffset, true);
                ctx.WriteCorner(job, corner + 3u, current_inner, current_source, current_source, 0.f, InvalidOffset, true);
                TopoEmitFace(ctx, job, face, corner, 4u, ctx.SrcFaceOf(job, current_source), true);
            }
        };
        const BevelRingInfo boundary = TopoBevelRing(ctx, job, v, [&](uint2 id, uint source, uint index) {
            if (index >= length) return; // A closed fan reports its first point twice before dropping the duplicate.
            if (index == 0u) { first_id = id; first_source = source; }
            else quad(id, source, index, previous_id, previous_source);
            for (uint ring = 1u; ring < segments; ++ring) {
                const uint d = TopoBevelVertexRingIndex(ctx, job, entry, id, index, length, ring);
                const float3 center = ctx.SrcPosition(job, v);
                ctx.WriteVertexMap(job, d, v, v, 0.f);
                *ctx.Inward(job, d, 0u) = packed_float3(TopoBevelVertexRingPosition(ctx, job, v, id, source, ring) - center);
                ctx.SelectDstVertex(job, d);
            }
            const uint cap = TopoBevelVertexRingIndex(ctx, job, entry, id, index, length, segments - 1u);
            ctx.WriteCorner(job, base + 4u * (segments - 1u) * length + length - 1u - index,
                cap, source, source, 0.f, InvalidOffset, true);
            previous_id = id; previous_source = source;
        });
        if (segments > 1u) quad(first_id, first_source, 0u, previous_id, previous_source);
        TopoEmitFace(ctx, job, fd + (segments - 1u) * length,
            base + 4u * (segments - 1u) * length, length, ctx.SrcFaceOf(job, boundary.FirstSource), true);
        return;
    }
    if (count == 0u) return;
    // The vertex's own output comes first, then its copy, and only the copy of an extrusion is selected.
    const uint d = offsets[entry];
    const bool kept = TopoVertexKept(ctx, job, v);
    const uint first_copy = d + (kept ? 1u : 0u);
    // A rip keeps the selection on the copies it tears away, and a plain split keeps it on every sector.
    const bool rip = job.Op == MeshTopologyOp::EdgeSplit && (job.Flags & TopologyFlagRipSelectCopies) != 0u;
    if (kept) {
        ctx.WriteVertexMap(job, d, v, v, 0.f);
        ctx.SelectDstVertex(job, d, TopoOriginalVertexSelected(ctx, job, v) && !(rip && count > 1u));
    }
    for (uint copy = first_copy; copy < d + count; ++copy) {
        ctx.WriteVertexMap(job, copy, v, v, 0.f);
        if ((!rip || copy == d + 1u) && (job.Op!=MeshTopologyOp::ExtrudeRegion || copy==d+count-1u)) ctx.SelectDstVertex(job, copy);
    }
    if (job.Op == MeshTopologyOp::Wireframe && first_copy < d + count) TopoWireVertices(ctx, job, v, first_copy);
    if (job.Op==MeshTopologyOp::ExtrudeRegion) {
        if (TopoRegionConnectsVertex(ctx,job,v)) {
            const uint corner=ctx.SrcAnyCornerAt(job,v), base=ctx.WireCornerOffset(job,entry);
            for (uint layer=1u;layer<=job.Steps;++layer)
                TopoEmitLine(ctx,job,base+2u*(layer-1u),d+layer-1u,d+layer,corner,corner,InvalidOffset,false);
        }
        if (TopoTransformsCopies(job) && (ctx.FlagVertices(job)[v]&TopoInRegion)) {
            const float3 p=ctx.SrcPosition(job,v);
            float3 q=p;
            for (uint layer=1u;layer<=job.Steps;++layer) {
                q=ctx.TransformCopy(job,q);
                if (layer==job.Steps || !TopoRegionHasSides(ctx,job) || (ctx.FlagVertices(job)[v]&TopoOnBoundary))
                    *ctx.Inward(job,TopoRegionLayerVertex(ctx,job,v,layer),0u)=packed_float3(q-p);
            }
        }
        return;
    }
    if (TopoLineCore(job) && !TopologyCopiesSelection(job.Op) && !TopologyExtrudesSides(job.Op) && first_copy < d + count) {
        const uint corner = ctx.SrcAnyCornerAt(job, v);
        TopoEmitLine(ctx, job, ctx.WireCornerOffset(job, entry), d, first_copy, corner, corner, InvalidOffset, false);
    }
    // A transform moves every vertex of the moved or duplicated faces: the copies, and the inner vertices a region moves in place.
    // A solidify pushes the copies in along the vertex normal.
    const bool transform = TopoTransformsCopies(job), solid = job.Op == MeshTopologyOp::Solidify;
    // Each layer of copies moves through the transform once more than the last.
    if (transform || solid) {
        const float3 p = ctx.SrcPosition(job, v);
        float3 q = p;
        for (uint copy = first_copy; copy < d + count; ++copy) {
            q = transform ? ctx.TransformCopy(job, q) : p - normalize(float3(ctx.SrcVertexNormals(job)[v])) * job.Param0;
            *ctx.Inward(job, copy, 0u) = packed_float3(q - p);
        }
        if (transform && kept && count == 1u && TopoRegionMoves(ctx, job) && (ctx.FlagVertices(job)[v] & TopoInRegion)) {
            for (uint layer = 0u; layer < job.Steps; ++layer) q = ctx.TransformCopy(job, q);
            *ctx.Inward(job, d, 0u) = packed_float3(q - p);
        }
    }
}

kernel void TopologyScatterHalfedges(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshTopologyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const TopoContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshTopologyJob job = ctx.Jobs()[tile.x];
    const uint hi = tile.y * ScanTileSize + lane;
    const uint h = ctx.SrcHalfedgeDomain(job).Handle(hi);
    if (h == InvalidOffset) return;
    const uint entry = ctx.HalfedgeEntry(job, h);
    if (TopologyIsDissolve(job.Op) && (TopoLineJoins(job) || ctx.Src(job).HalfedgeFace(h)==InvalidOffset)) {
        const bool wire=ctx.Src(job).HalfedgeFace(h)==InvalidOffset;
        const bool kept=ctx.Counts(job,TopoCountWireCorners)[entry+1u]!=ctx.Counts(job,TopoCountWireCorners)[entry];
        if (kept || (wire && ctx.SrcEdgeFirst(job,h)))
            ctx.WireEdgeMap(job)[ctx.SrcEdge(job,h)]=kept ? job.DstCornerOffset+ctx.WireCornerOffset(job,entry)+1u : InvalidOffset;
        if (kept) TopoScatterLine(ctx,job,h,entry);
        return;
    }
    if (TopologyIsMerge(job.Op)) {
        if (!ctx.SrcEdgeFirst(job,h)) return;
        const auto line=TopoLineOutput(ctx,job,h);
        uint mapped=InvalidOffset;
        if (line.Kept) {
            const uint rep=TopoLineRepresentative(ctx,job,line);
            const uint corner=ctx.SrcHalfedgeDomain(job).Handle(rep);
            mapped=TopoLineSurface(ctx,job,corner) ? 0u : job.DstCornerOffset+ctx.WireCornerOffset(job,ctx.HalfedgeEntry(job,corner))+1u;
            TopoScatterLine(ctx,job,h,entry);
        }
        ctx.WireEdgeMap(job)[ctx.SrcEdge(job,h)]=mapped;
        return;
    }
    // Surface edges retain surface identity; rewritten wires match their emitted pair and endpoints.
    if ((TopologyExtrudesSides(job.Op) || job.Op==MeshTopologyOp::Subdivide || TopologyCopiesSelection(job.Op)) && hi<job.SrcEdgeCount &&
        ctx.Src(job).HalfedgeFace(ctx.SrcEdgeHalfedge(job,hi))!=InvalidOffset) ctx.WireEdgeMap(job)[hi]=0u;
    if (TopologyCopiesSelection(job.Op)) {
        const uint count=ctx.Counts(job,TopoCountWireCorners)[entry+1u]-ctx.Counts(job,TopoCountWireCorners)[entry];
        if (!count) return;
        TopoScatterLine(ctx, job, h, entry);
        if (ctx.Src(job).HalfedgeFace(h) == InvalidOffset)
            ctx.WireEdgeMap(job)[ctx.SrcEdge(job, h)] = job.DstCornerOffset + ctx.WireCornerOffset(job, entry) + 1u;
        return;
    }
    if (TopologyIsDelete(job.Op)) {
        const uint edge=ctx.SrcEdge(job,h);
        if (ctx.SrcEdgeFirst(job,h) && TopoEdgeDeleted(ctx,job,h)) ctx.WireEdgeMap(job)[edge]=InvalidOffset;
        if (ctx.Counts(job,TopoCountWireCorners)[entry+1u]!=ctx.Counts(job,TopoCountWireCorners)[entry]) {
            TopoScatterLine(ctx, job, h, entry);
            ctx.WireEdgeMap(job)[edge] = job.DstCornerOffset + ctx.WireCornerOffset(job, entry) + 1u;
        }
        return;
    }
    if (job.Op==MeshTopologyOp::AddPrimitives && ctx.Src(job).HalfedgeFace(h)==InvalidOffset) return;
    if (TopologyExtrudesSides(job.Op) && ctx.SrcEdgeFirst(job,h) && ctx.Src(job).HalfedgeFace(h)==InvalidOffset) {
        const uint e=ctx.SrcEdge(job,h);
        ctx.WireEdgeMap(job)[e]=0u;
        if (!TopoHalfedgeMakesSide(ctx,job,h)) {
            TopoScatterLine(ctx, job, h, entry);
            ctx.WireEdgeMap(job)[e] = job.DstCornerOffset + ctx.WireCornerOffset(job, entry) + 1u;
            return;
        }
    }
    if ((TopoLineCore(job) && !TopologyExtrudesSides(job.Op)) ||
        ((job.Op==MeshTopologyOp::Subdivide || job.Op==MeshTopologyOp::KeepSelectedFaces) && ctx.Src(job).HalfedgeFace(h)==InvalidOffset)) {
        if (ctx.SrcEdgeFirst(job, h)) {
            TopoScatterLine(ctx, job, h, entry);
            if (job.Op==MeshTopologyOp::Subdivide)
                ctx.WireEdgeMap(job)[ctx.SrcEdge(job,h)]=TopoEdgeCut(ctx,job,h) ? InvalidOffset : job.DstCornerOffset+ctx.WireCornerOffset(job,entry)+1u;
        }
        return;
    }
    if (TopologyIsBevel(job.Op)) {
        const BevelHalfedgeOutputs o = TopoBevelOutputs(ctx, job, h);
        const auto corners = ctx.SrcCorners(job);
        const uint v = corners[h], u = corners[ctx.SrcPrev(job, h)];
        const uint base = ctx.Counts(job, TopoCountVertices)[entry];
        // Every new vertex sits on an original one, displaced to its point.
        const auto place = [&](uint d, uint anchor, float3 p) {
            ctx.WriteVertexMap(job, d, anchor, anchor, 0.f);
            *ctx.Inward(job, d, 0u) = packed_float3(p - ctx.SrcPosition(job, anchor));
            ctx.SelectDstVertex(job, d);
        };
        float3 point;
        if (o.EndFrom && TopoBevelEdgePoint(ctx, job, h, false, point)) place(base + o.EndIndex(false), u, point);
        if (o.EndTo && TopoBevelEdgePoint(ctx, job, h, true, point)) place(base + o.EndIndex(true), v, point);
        if (o.Corner) place(base + o.CornerIndex(), v, TopoBevelCornerPoint(ctx, job, h));
        if (o.Profiles == 0u) {
            if (!(ctx.SrcEdgeFirst(job, h) && TopoEdgeBeveled(ctx, job, h))) return;
        }
        // A beveled representative's strip: rows from its own face's side to its opposite's, through the profile points.
        const uint opposite = ctx.SrcOpposite(job, h);
        const uint segments = TopoBevelSegments(job);
        const float3 side_from_a = TopoBevelSidePosition(ctx, job, h, false), side_to_a = TopoBevelSidePosition(ctx, job, h, true);
        float3 side_from_b = side_from_a, side_to_b = side_to_a;
        if (opposite != InvalidOffset) {
            side_from_b = TopoBevelSidePosition(ctx, job, opposite, true);
            side_to_b = TopoBevelSidePosition(ctx, job, opposite, false);
        }
        const float3 pu = ctx.SrcPosition(job, u), pv = ctx.SrcPosition(job, v);
        for (uint j = 1u; j < segments; ++j) {
            const float t = float(j) / float(segments), a = (1.f - t) * (1.f - t), b = 2.f * (1.f - t) * t, c = t * t;
            place(base + o.ProfileIndex(false, j - 1u, segments), u, side_from_a * a + pu * b + side_from_b * c);
            place(base + o.ProfileIndex(true, j - 1u, segments), v, side_to_a * a + pv * b + side_to_b * c);
        }
        device const uint *face_offsets = ctx.Counts(job, TopoCountFaces);
        uint fd = face_offsets[entry];
        uint corner_base = ctx.Counts(job, TopoCountCorners)[entry];
        const uint side_u_a = TopoBevelVertexOf(ctx, job, TopoBevelSideVertex(ctx, job, h, false)), side_v_a = TopoBevelVertexOf(ctx, job, TopoBevelSideVertex(ctx, job, h, true));
        const uint side_u_b = opposite != InvalidOffset ? TopoBevelVertexOf(ctx, job, TopoBevelSideVertex(ctx, job, opposite, true)) : side_u_a;
        const uint side_v_b = opposite != InvalidOffset ? TopoBevelVertexOf(ctx, job, TopoBevelSideVertex(ctx, job, opposite, false)) : side_v_a;
        const uint corner_u_a = ctx.SrcPrev(job, h), corner_v_a = h;
        const uint corner_u_b = opposite != InvalidOffset ? opposite : corner_u_a, corner_v_b = opposite != InvalidOffset ? ctx.SrcPrev(job, opposite) : corner_v_a;
        const auto row_u = [&](uint j) { return j == 0u ? side_u_a : j == segments ? side_u_b : base + o.ProfileIndex(false, j - 1u, segments); };
        const auto row_v = [&](uint j) { return j == 0u ? side_v_a : j == segments ? side_v_b : base + o.ProfileIndex(true, j - 1u, segments); };
        for (uint j = 0u; j < segments; ++j) {
            const uint4 loop = uint4(row_v(j), row_u(j), row_u(j + 1u), row_v(j + 1u));
            const float t0 = float(j) / float(segments), t1 = float(j + 1u) / float(segments);
            const uint4 source_a = uint4(corner_v_a, corner_u_a, corner_u_a, corner_v_a), source_b = uint4(corner_v_b, corner_u_b, corner_u_b, corner_v_b);
            const float4 weights = float4(t0, t0, t1, t1);
            for (uint k = 0u; k < 4u; ++k) ctx.WriteCorner(job, corner_base + k, loop[k], source_a[k], source_b[k], weights[k], InvalidOffset, true);
            TopoEmitFace(ctx, job, fd, corner_base, 4u, ctx.SrcFaceOf(job, h), true);
            ++fd;
            corner_base += 4u;
        }
        return;
    }
    if (job.Op == MeshTopologyOp::Subdivide) {
        // A selected edge's cuts sit along its representative halfedge, each selected.
        device const uint *vertex_offsets = ctx.Counts(job, TopoCountVertices);
        const uint first = vertex_offsets[entry], count = vertex_offsets[entry + 1u] - first;
        if (count == 0u) return;
        const auto corners = ctx.SrcCorners(job);
        const uint from = corners[ctx.SrcPrev(job, h)], to = corners[h];
        for (uint i = 0u; i < count; ++i) {
            ctx.WriteVertexMap(job, first + i, from, to, TopoCutParam(ctx, job, h, i, count));
            ctx.SelectDstVertex(job, first + i);
        }
        return;
    }
    device const uint *face_offsets = ctx.Counts(job, TopoCountFaces);
    if (face_offsets[entry + 1u] == face_offsets[entry]) return;
    const uint fd = face_offsets[entry];
    const uint base = ctx.Counts(job, TopoCountCorners)[entry];
    device const uint *vertex_offsets = ctx.Counts(job, TopoCountVertices);
    const auto corners = ctx.SrcCorners(job);
    const uint prev = ctx.SrcPrev(job, h);
    const uint a = corners[prev], b = corners[h];
    // The quad winds to match the face left beside it.
    const bool flip = job.Op==MeshTopologyOp::ExtrudeRegion ? TopoRegionEdgeFaces(ctx,job,ctx.SrcEdge(job,h))==0u :
        (ctx.FlagHalfedges(job)[h] & TopoSideFlip) != 0u || job.Op == MeshTopologyOp::Solidify;
    const uint a_out = vertex_offsets[a], b_out = vertex_offsets[b];
    const uint4 sources = flip ? uint4(h, prev, prev, h) : uint4(prev, h, h, prev);
    if (job.Op == MeshTopologyOp::InsetRegion) {
        // The boundary edge's inward direction inside its selected face reaches both copies: one arriving, one leaving.
        const float3 n = normalize(float3(ctx.SrcFaceNormals(job)[ctx.SrcFaceOf(job, h)]));
        const float3 inward = normalize(cross(n, ctx.SrcPosition(job, b) - ctx.SrcPosition(job, a)));
        *ctx.Inward(job, b_out + 1u, 0u) = packed_float3(inward);
        *ctx.Inward(job, a_out + 1u, 1u) = packed_float3(inward);
    }
    // One quad per layer, from the layer below to the layer's copies, and only the last layer's top edge is selected.
    const uint bottom = 1u, top = 3u;
    for (uint layer = 1u; layer <= job.Steps; ++layer) {
        const uint a_below = a_out + layer - 1u, b_below = b_out + layer - 1u, a_copy = a_out + layer, b_copy = b_out + layer;
        const uint4 loop = flip ? uint4(b_below, a_below, a_copy, b_copy) : uint4(a_below, b_below, b_copy, a_copy);
        const uint quad = base + 4u * (layer - 1u);
        for (uint k = 0u; k < 4u; ++k) ctx.WriteCorner(job, quad + k, loop[k], sources[k], sources[k], 0.f, k == bottom || k == top ? h : InvalidOffset, k == top && layer == job.Steps);
        TopoEmitFace(ctx, job, fd + layer - 1u, quad, 4u, ctx.SrcFaceOf(job, h), false);
    }
}

// Emits each source face's outputs: its own loop through the vertex targets and copies, a duplicate loop, or an individual extrusion.
kernel void TopologyScatterFaces(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshTopologyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const TopoContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshTopologyJob job = ctx.Jobs()[tile.x];
    const uint f = tile.y * ScanTileSize + lane;
    if (f > job.SrcFaceCount) return;
    const uint entry = ctx.FaceEntry(job, f);
    if (ctx.Counts(job, TopoCountFaces)[entry + 1u] == ctx.Counts(job, TopoCountFaces)[entry] &&
        ctx.Counts(job, TopoCountWireCorners)[entry + 1u] == ctx.Counts(job, TopoCountWireCorners)[entry]) return;
    uint fd = ctx.Counts(job, TopoCountFaces)[entry];
    uint base = ctx.Counts(job, TopoCountCorners)[entry];
    const uint new_vertices = ctx.Counts(job, TopoCountVertices)[entry];
    const bool list_entry = f == job.SrcFaceCount;
    const uint2 range = list_entry ? uint2(0u, 0u) : ctx.SrcFaceRange(job, f);
    const uint n = range.y - range.x;
    const bool selected = !list_entry && ctx.SrcSelectedFace(job, f);
    device const uint *vertex_offsets = ctx.Counts(job, TopoCountVertices);
    const auto src_corners = ctx.SrcCorners(job);
    device const uint *targets = ctx.VertexTargets(job);
    if (TopologyIsMerge(job.Op) && !list_entry) {
        TopoEmitWeld(ctx,job,f,fd,base);
        return;
    }
    // Original and copied loops share winding, edge provenance and face publication.
    const auto source_loop = [&](uint layer, bool reverse, bool select, bool keep_edge_selection) {
        for (uint k = 0u; k < n; ++k) {
            const uint h = range.x + (reverse ? n - 1u - k : k), v = src_corners[h];
            const uint edge = reverse ? ctx.SrcNext(job, h) : h;
            const uint output = job.Op == MeshTopologyOp::EdgeSplit ? TopoSectorVertex(ctx, job, h) :
                job.Op == MeshTopologyOp::ExtrudeRegion && layer ? TopoRegionLayerVertex(ctx, job, v, layer) : vertex_offsets[v] + layer;
            ctx.WriteCorner(job, base + k, output, h, h, 0.f, edge,
                keep_edge_selection ? ctx.SrcSelectedEdge(job, ctx.SrcEdge(job, edge)) : select);
        }
        TopoEmitFace(ctx, job, fd++, base, n, f, select);
        base += n;
    };
    if (job.Op == MeshTopologyOp::ExtrudeRegion && !list_entry) {
        if (!selected || !ctx.DelOrig(job)) {
            source_loop(0u, selected, false, false);
        }
        if (selected) {
            const uint first=TopoRegionHasSides(ctx,job) ? job.Steps : 1u;
            for (uint layer=first;layer<=job.Steps;++layer) {
                source_loop(layer, layer != job.Steps, layer == job.Steps, false);
            }
        }
        return;
    }
    if (job.Op == MeshTopologyOp::ReplaceFaces) {
        const auto list=ctx.Lists(job);
        const auto record=list+list[2u+list[0]+f];
        uint cursor=2u;
        for (uint j=0u;j<record[1];++j) {
            const uint count=record[cursor++];
            for (uint i=0u;i<count;++i) {
                const uint h=record[cursor++],edge=record[cursor++];
                ctx.WriteCorner(job,base+i,vertex_offsets[src_corners[h]],h,h,0.f,edge,ctx.SrcSelectedEdge(job,ctx.SrcEdge(job,edge)));
            }
            TopoEmitFace(ctx,job,fd++,base,count,f,selected); base+=count;
        }
        return;
    }
    if ((job.Op == MeshTopologyOp::SplitNonplanarFaces || job.Op == MeshTopologyOp::SplitConcaveFaces) && selected && n > 3u) {
        TopoFaceSplitEmit(ctx, job, f, fd, base);
        return;
    }
    if (job.Op == MeshTopologyOp::Wireframe && selected) {
        TopoWireFace(ctx, job, f, fd, base, new_vertices);
        return;
    }
    if (job.Op == MeshTopologyOp::Subdivide) {
        SubdivideEmitter emitter{ctx, job, f, true, fd, base, 0u, 0u};
        TopoSubdivideFace(ctx, job, f, emitter);
        return;
    }
    if (job.Op == MeshTopologyOp::ConnectVertices || job.Op == MeshTopologyOp::RotateEdges) {
        const bool own_loop = job.Op == MeshTopologyOp::ConnectVertices || TopoDissolveOwnLoop(ctx, job, f);
        if (own_loop || ctx.FaceLabels(job)[f] == f) TopoEmitConnect(ctx, job, f, own_loop, TopoMeasureConnect(ctx, job, f, own_loop), fd, base);
        return;
    }
    if (TopologyIsBevel(job.Op)) {
        // Each corner emits its replacement points. A corner replaced by one point keeps its arriving edge, and a corner replaced by two keeps neither.
        uint emitted = 0u;
        for (uint h = range.x; h < range.y; ++h) {
            const BevelCornerPoints points = TopoBevelCornerPoints(ctx, job, h);
            const bool edge_selected = ctx.SrcSelectedEdge(job, ctx.SrcEdge(job, h)) && !TopoEdgeBeveled(ctx, job, h);
            for (uint i = 0u; i < points.Count; ++i) {
                ctx.WriteCorner(job, base + emitted++, TopoBevelVertexOf(ctx, job, points.Vertex[i]), h, h, 0.f, i == 0u && points.Count == 1u ? h : InvalidOffset, edge_selected);
            }
        }
        TopoEmitFace(ctx, job, fd, base, emitted, f, selected);
        return;
    }
    if (job.Op == MeshTopologyOp::Triangulate && selected && n > 3u) {
        TopoTriangulateFace(ctx, job, f, [&](uint3 tri, uint t) {
            const uint corner_index[3] = {tri.x, tri.y, tri.z};
            for (uint q = 0u; q < 3u; ++q) {
                const uint k = corner_index[q], from = corner_index[(q + 2u) % 3u];
                const uint h = range.x + k;
                // A triangle side along the face loop keeps that edge, and a diagonal is new.
                const bool along = (from + 1u) % n == k;
                ctx.WriteCorner(job, base + q, vertex_offsets[src_corners[h]], h, h, 0.f, along ? h : InvalidOffset, true);
            }
            TopoEmitFace(ctx, job, fd + t, base, 3u, f, true);
            base += 3u;
        });
        return;
    }
    if (job.Op == MeshTopologyOp::TrisToQuads && ctx.FaceLabels(job)[f] != InvalidOffset) {
        const uint g = ctx.FaceLabels(job)[f];
        if (g < f) return;
        // The quad runs p, q, s, r: this triangle's corners with the partner's third vertex between the shared edge's ends.
        const uint h = ctx.RegionStart(job)[f], opposite = ctx.SrcOpposite(job, h);
        const uint hp = ctx.SrcNext(job, h), hq = ctx.SrcPrev(job, h), hs = ctx.SrcNext(job, opposite), hr_partner = ctx.SrcNext(job, hs);
        const uint4 corners_out = uint4(hp, hq, hs, h);
        // Each side keeps the edge it came from, read at its arriving corner.
        const uint4 edges = uint4(hp, hq, hs, hr_partner);
        for (uint q = 0u; q < 4u; ++q) {
            ctx.WriteCorner(job, base + q, vertex_offsets[src_corners[corners_out[q]]], corners_out[q], corners_out[q], 0.f, edges[q], ctx.SrcSelectedEdge(job, ctx.SrcEdge(job, edges[q])));
        }
        TopoEmitFace(ctx, job, fd, base, 4u, f, true);
        return;
    }
    if (job.Op == MeshTopologyOp::Poke && selected) {
        // The center vertex averages the face's corners and lifts by the offset along the face normal.
        const uint center = new_vertices;
        const uint h0 = range.x, h1 = range.x + (n >= 4u ? n / 4u : 1u), h2 = range.x + (n >= 4u ? n / 2u : 2u), h3 = range.x + (n >= 4u ? (3u * n) / 4u : 2u);
        const float t = n >= 4u ? 0.5f : 1.f / 3.f;
        ctx.WriteVertexMap4(job, center, uint4(src_corners[h0], src_corners[h1], src_corners[h2], src_corners[h3]), 0.5f, t);
        const float3 normal = float3(ctx.SrcFaceNormals(job)[f]);
        const float normal_length2 = dot(normal, normal);
        *ctx.Inward(job, center, 0u) = packed_float3(job.Param0 != 0.f && normal_length2 > 0.f ? normal * (job.Param0 * rsqrt(normal_length2)) : float3(0));
        ctx.SelectDstVertex(job, center);
        for (uint k = 0u; k < n; ++k) {
            const uint h = range.x + k, next = range.x + (k + 1u) % n;
            ctx.WriteCorner(job, base, vertex_offsets[src_corners[h]], h, h, 0.f, InvalidOffset, true);
            ctx.WriteCorner(job, base + 1u, vertex_offsets[src_corners[next]], next, next, 0.f, next, ctx.SrcSelectedEdge(job, ctx.SrcEdge(job, next)));
            ctx.WriteCorner4(job, base + 2u, center, uint4(h0, h1, h2, h3), 0.5f, t, InvalidOffset, true);
            TopoEmitFace(ctx, job, fd + k, base, 3u, f, true);
            base += 3u;
        }
        return;
    }
    if ((job.Op == MeshTopologyOp::FlipNormals && selected) || job.Op == MeshTopologyOp::EdgeSplit) {
        source_loop(0u, job.Op == MeshTopologyOp::FlipNormals, selected, true);
        return;
    }
    if (TopologyIsDissolve(job.Op) && !TopoDissolveOwnLoop(ctx, job, f)) {
        // A region root emits its walked boundary as one face, selected when the face was selected or the region spans more faces.
        TopoWalkRegion(ctx, job, f, fd, base);
        TopoEmitFace(ctx, job, fd, base, ctx.WalkLength(job)[f], f, selected || ctx.RegionBoundary(job)[f] != n);
        return;
    }
    const bool individual = TopologyBaseOp(job.Op) == MeshTopologyOp::ExtrudeFacesIndividual && selected;
    const bool duplicated = selected && TopoRegionDuplicates(ctx, job);
    // A deletion, merge, or keep leaves every surviving edge's selection alone, and an extrusion selects only the moved faces' edges.
    const bool edges_from_source = job.Op == MeshTopologyOp::DeleteVertices || job.Op == MeshTopologyOp::DeleteEdges ||
        job.Op == MeshTopologyOp::DeleteFaces || job.Op == MeshTopologyOp::DeleteOnlyEdgesFaces ||
        job.Op == MeshTopologyOp::DeleteOnlyFaces || job.Op == MeshTopologyOp::KeepSelectedFaces ||
        job.Op == MeshTopologyOp::AddPrimitives ||
        TopologyIsDissolve(job.Op) || TopologyIsMerge(job.Op);
    const bool own_selected = individual || duplicated ? false : (job.Op == MeshTopologyOp::KeepSelectedFaces || selected) && job.Op != MeshTopologyOp::ExtrudeEdges;
    // The list entry appends the job's listed faces, whose corners inherit attributes from any corner at their vertex.
    if (f == job.SrcFaceCount) {
        if (job.Op != MeshTopologyOp::AddPrimitives || job.ListOffset == InvalidOffset) return;
        device const uint *list = ctx.Lists(job);
        // New vertices inherit attributes from a vertex explicitly named by
        // the face list, so local emission has no unrelated source dependency.
        const uint listed = list[0];
        const uint attribute_source = list[3];
        const uint span=list[2],boundary=list[1];
        const uint4 grid_vertices=listed ? uint4(list[TopologyPrimitiveHeaderWords],list[TopologyPrimitiveHeaderWords+span],
            list[TopologyPrimitiveHeaderWords+boundary/2u],list[TopologyPrimitiveHeaderWords+boundary/2u+span]) : uint4(attribute_source);
        uint4 grid_corners=uint4(InvalidOffset);
        if (listed) for (uint k=0u;k<4u;++k) grid_corners[k]=ctx.SrcAnyCornerAt(job,grid_vertices[k]);
        for (uint i = 0u; i < listed; ++i) {
            const uint length=list[1],s=list[2],t=length/2u-s;
            const uint x=i%(s-1u)+1u,y=i/(s-1u)+1u;
            const float u=float(x)/float(s),v=float(y)/float(t);
            const auto rail=[&](uint k) { return ctx.SrcPosition(job,list[TopologyPrimitiveHeaderWords+k%length]); };
            const float3 p=rail(x)*(1.f-v)+rail(2u*s+t-x)*v+rail(s+y)*u+rail(length-y)*(1.f-u)-
                (rail(0u)*((1.f-u)*(1.f-v))+rail(s)*(u*(1.f-v))+rail(2u*s+t)*((1.f-u)*v)+rail(s+t)*(u*v));
            ctx.WriteVertexMap4(job,new_vertices+i,grid_vertices,u,v);
            const float3 base=TopoBilinear(ctx.SrcPosition(job,grid_vertices.x),ctx.SrcPosition(job,grid_vertices.y),
                ctx.SrcPosition(job,grid_vertices.z),ctx.SrcPosition(job,grid_vertices.w),u,v);
            *ctx.Inward(job, new_vertices + i, 0u) = packed_float3(p-base);
            ctx.SelectDstVertex(job, new_vertices + i);
        }
        uint cursor = TopologyPrimitiveHeaderWords + list[1];
        const uint faces = list[cursor++];
        uint wire_base=ctx.WireCornerOffset(job,ctx.FaceEntry(job,f));
        for (uint i = 0u; i < faces; ++i) {
            const uint length = list[cursor++];
            if (length==2u) {
                const uint a=list[cursor], b=list[cursor+2u];
                TopoEmitLine(ctx,job,wire_base,a<job.SrcVertexCount ? vertex_offsets[a] : new_vertices+a-job.SrcVertexCount,
                    b<job.SrcVertexCount ? vertex_offsets[b] : new_vertices+b-job.SrcVertexCount,InvalidOffset,InvalidOffset,InvalidOffset,true);
                wire_base+=2u; cursor+=4u;
                continue;
            }
            // A listed vertex's corner inherits from the first source vertex in its face.
            uint any = job.SrcFaceCount ? ctx.SrcAnyCornerAt(job, attribute_source) : InvalidOffset;
            uint source_face = job.SrcFaceCount ? ctx.SrcFaceOf(job, any) : InvalidOffset;
            for (uint k = 0u; k < length; ++k) {
                if (list[cursor + 2u*k] < job.SrcVertexCount) {
                    any = ctx.SrcAnyCornerAt(job, list[cursor + 2u*k]);
                    if (job.SrcFaceCount) {
                        const uint incident = ctx.SrcFaceOf(job, any);
                        if (incident != InvalidOffset) source_face = incident;
                    }
                    break;
                }
            }
            for (uint k = 0u; k < length; ++k) {
                const uint v = list[cursor + 2u*k];
                const uint source_corner = v < job.SrcVertexCount ? ctx.SrcAnyCornerAt(job, v) : any;
                if (v<job.SrcVertexCount) ctx.WriteCorner(job,base+k,vertex_offsets[v],source_corner,source_corner,0.f,list[cursor+2u*k+1u],true);
                else {
                    const uint i=v-job.SrcVertexCount,x=i%(span-1u)+1u,y=i/(span-1u)+1u;
                    ctx.WriteCorner4(job,base+k,new_vertices+i,grid_corners,float(x)/float(span),float(y)/float(boundary/2u-span),list[cursor+2u*k+1u],true);
                }
            }
            // Local meshlet repair needs an adjacent source partition.
            TopoEmitFace(ctx, job, fd, base, length, source_face, true);
            ++fd;
            base += length;
            cursor += 2u*length;
        }
        return;
    }
    if (individual) {
        // The face's own copies of its corner vertices carry the top face, and each edge gets a quad down to the original vertices.
        for (uint k = 0u; k < n; ++k) {
            const uint h = range.x + k;
            const uint copy = new_vertices + k;
            ctx.WriteVertexMap(job, copy, src_corners[h], src_corners[h], 0.f);
            ctx.SelectDstVertex(job, copy);
            if (job.Op == MeshTopologyOp::InsetIndividual) {
                const uint prev = range.x + (k + n - 1u) % n, next = range.x + (k + 1u) % n;
                const float3 normal = normalize(float3(ctx.SrcFaceNormals(job)[f]));
                const float3 width = TopoInsetCorner(ctx.SrcPosition(job, src_corners[prev]), ctx.SrcPosition(job, src_corners[h]), ctx.SrcPosition(job, src_corners[next]), normal, 1.f, 0.f, (job.Flags & TopologyFlagEvenOffset) != 0u);
                *ctx.Inward(job, copy, 0u) = packed_float3(width);
                *ctx.Inward(job, copy, 1u) = packed_float3(normal);
            }
            ctx.WriteCorner(job, base + k, copy, h, h, 0.f, h, true);
        }
        TopoEmitFace(ctx, job, fd, base, n, f, true);
        ++fd;
        base += n;
        for (uint k = 0u; k < n; ++k) {
            const uint h = range.x + k, prev = k == 0u ? range.y - 1u : h - 1u;
            const uint a = vertex_offsets[src_corners[prev]], b = vertex_offsets[src_corners[h]];
            const uint4 loop = uint4(a, b, new_vertices + k, new_vertices + (k == 0u ? n - 1u : k - 1u));
            const uint4 sources = uint4(prev, h, h, prev);
            for (uint j = 0u; j < 4u; ++j) ctx.WriteCorner(job, base + j, loop[j], sources[j], sources[j], 0.f, j == 1u || j == 3u ? h : InvalidOffset, j == 3u);
            TopoEmitFace(ctx, job, fd, base, 4u, f, false);
            ++fd;
            base += 4u;
        }
        return;
    }
    // The face's own loop: a run of corners mapping to one vertex keeps its first corner, whose halfedge arrives from the previous run.
    const uint length = TopoMappedLoopLength(ctx, job, f);
    uint emitted = 0u, previous = InvalidOffset;
    for (uint h = range.x; h < range.y && emitted < length; ++h) {
        const uint v = src_corners[h];
        if (TopoVertexRemoved(ctx, job, v)) continue;
        const uint m = targets[v];
        if (m == previous) continue;
        previous = m;
        const uint v_out = vertex_offsets[m] + (TopoFaceUsesCopy(ctx, job, selected, v) ? job.Steps : 0u);
        const bool listed_edge=job.Op==MeshTopologyOp::AddPrimitives && (ctx.FlagHalfedges(job)[h]&TopoListed);
        ctx.WriteCorner(job, base + emitted++, v_out, h, h, 0.f, h, listed_edge || ((edges_from_source || own_selected) && ctx.SrcSelectedEdge(job, ctx.SrcEdge(job, h))));
    }
    TopoEmitFace(ctx, job, fd, base, length, f, own_selected && selected);
    if (!duplicated) return;
    // Each layer's copies keep the original winding unless the copy transform flips it.
    ++fd;
    base += n;
    for (uint layer = 1u; layer <= job.Steps; ++layer) source_loop(layer, TopoFlipsCopies(job), true, false);
}

kernel void TopologyFaceTables(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshTopologyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const TopoContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshTopologyJob job = ctx.Jobs()[tile.x];
    const uint fd = tile.y * ScanTileSize + lane;
    if (fd >= job.DstFaceCount) return;
    // A polygon has two fewer triangles than corners, independent of its diagonals.
    const uint2 range = ctx.DstFaceRange(job, fd);
    const uint first = range.x - 2u * fd;
    ctx.DstFaceTriangles(job)[fd] = job.DstTriangleOffset + first;
    device packed_uint3 *triangles = ctx.DstTriangles(job);
    const uint corner = job.DstCornerOffset + range.x;
    const uint source = ctx.FaceMap(job)[fd];
    uint source_triangle = InvalidOffset;
    device uint *triangle_sources = job.DstTriangleSources.Slot != InvalidSlot ?
        BindlessBufferMutable(uint,bindless.Buffer,job.DstTriangleSources.Slot) + job.DstTriangleSources.Offset : nullptr;
    if (triangle_sources && source != InvalidOffset) {
        source_triangle = BindlessBuffer(uint,bindless.ObjectIdBuffer,pc.Source.FaceTriangleStartSlot)[ctx.SrcFaceDomain(job).Handle(source)];
    }
    // Vertex gathering has finished; its provenance scratch now holds tessellation work.
    device uint *scratch=ctx.VertexMap(job);
    device uint *next=scratch+range.x,*previous=scratch+job.DstHalfedgeCount+range.x;
    device vec2 *points=reinterpret_cast<device vec2 *>(scratch+2u*job.DstHalfedgeCount)+range.x;
    const auto vertices=BindlessBuffer(Vertex,bindless.VertexBuffer,pc.Destination.VertexSlot);
    TriangulatePolygon(range.y-range.x,[&](uint i) { return vec3(vertices[ctx.DstCorners(job)[range.x+i]].Position); },
        points,next,previous,[&](uvec3 triangle,uint t) {
            triangles[first+t]=packed_uint3(uint3(triangle)+corner);
            if (triangle_sources) triangle_sources[first+t]=source_triangle;
        });
    const bool listed = job.Op == MeshTopologyOp::AddPrimitives && fd >= job.SrcFaceCount;
    const uint primitive = source != InvalidOffset ? ctx.SrcElementPrimitive(job, source) : 0u;
    ctx.SetDstElementPrimitive(job, fd, job.PrimitiveWork.Storage.Slot != InvalidSlot ? WorkRank(bindless, job.PrimitiveWork, primitive) : primitive);
    ctx.DstFaceSharpness(job)[fd] = !listed && source != InvalidOffset ? ctx.SrcFaceSharpness(job)[source] : uchar(0);
}

// Copy transforms carry authored normal directions through each generated layer.
inline float3 TopoCopyNormal(TopoContext ctx, MeshTopologyJob job, uint v, uint d, float3 normal) {
    if (!TopoTransformsCopies(job)) return normal;
    const uint first = ctx.Counts(job, TopoCountVertices)[v];
    const uint end = ctx.Counts(job, TopoCountVertices)[v + 1u];
    const bool kept = TopoVertexKept(ctx, job, v);
    const uint first_copy = first + uint(kept);
    uint steps = d >= first_copy && d < end ? d - first_copy + 1u : 0u;
    if (job.Op==MeshTopologyOp::ExtrudeRegion && steps && TopoRegionHasSides(ctx,job) && !(ctx.FlagVertices(job)[v]&TopoOnBoundary)) steps=job.Steps;
    if (d == first && kept && end == first + 1u && TopoRegionMoves(ctx, job) && (ctx.FlagVertices(job)[v] & TopoInRegion)) steps = job.Steps;
    const float3x3 m = job.CopyRotation.Unpack();
    const float3x3 cof{cross(m[1], m[2]), cross(m[2], m[0]), cross(m[0], m[1])};
    for (uint step = 0u; step < steps; ++step) normal = NormalizeOrZero(cof * normal);
    return normal;
}

// Carries every vertex-domain channel through the vertex map.
kernel void TopologyGatherVertices(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshTopologyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const TopoContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshTopologyJob job = ctx.Jobs()[tile.x];
    const uint d = tile.y * ScanTileSize + lane;
    if (d >= job.DstVertexCount) return;
    device const uint *map = ctx.VertexMap(job) + TopoVertexMapWords * d;
    const uint4 v = uint4(map[0], map[1], map[2], map[3]);
    const float s = as_type<float>(map[4]), t = as_type<float>(map[5]);
    const float4 weights=float4((1.f-s)*(1.f-t),s*(1.f-t),s*t,(1.f-s)*t);
    bool hidden=job.EditorState;
    for (uint k=0u;job.EditorState && k<4u;++k) if (weights[k]!=0.f)
        hidden=hidden && v[k]<job.SrcVertexCount && ctx.Selected({pc.Source.VertexHiddenSlot,0u},ctx.SrcVertexDomain(job).Handle(v[k]));
    if (job.EditorState) ctx.Select({pc.Destination.VertexHiddenSlot,0u},ctx.DstVertexDomain(job).Handle(d),hidden);
    if (hidden) ctx.SelectDstVertex(job,d,false);
    if (job.Op != MeshTopologyOp::MergeCollapse || !ctx.SrcSelectedVertex(job, v.x)) {
        const auto src = ctx.SrcVertices(job);
        const bool merged = TopologyIsMerge(job.Op) && v.x == job.TargetVertex;
        const float3 base = merged ? float3(job.TargetPosition) : TopoBilinear(float3(src[v.x].Position), float3(src[v.y].Position), float3(src[v.z].Position), float3(src[v.w].Position), s, t);
        float3 position = base, width = float3(0.f), depth = float3(0.f);
        if (job.Op == MeshTopologyOp::InsetRegion) {
            // A boundary copy sits between two inward directions, and an even offset keeps the inset width across the corner.
            const float3 in_a = float3(*ctx.Inward(job, d, 0u)), in_b = float3(*ctx.Inward(job, d, 1u));
            const bool has_a = dot(in_a, in_a) > 0.f, has_b = dot(in_b, in_b) > 0.f;
            if (has_a || has_b) {
                const float3 sum = in_a + in_b;
                const float scale = has_a && has_b ? ((job.Flags & TopologyFlagEvenOffset) != 0u ? 1.f / max(1.f + dot(in_a, in_b), 1e-4f) : 0.5f) : 1.f;
                width = sum * scale;
                depth = float3(ctx.SrcVertexNormals(job)[v.x]);
            }
        } else if (job.Op == MeshTopologyOp::InsetIndividual) {
            width = float3(*ctx.Inward(job, d, 0u));
            depth = float3(*ctx.Inward(job, d, 1u));
        } else if (job.Op==MeshTopologyOp::Decimate) {
            position=float3(*ctx.Inward(job,v.x,0u));
        } else if (TopologyDisplaces(job.Op, job.Flags)) {
            position += float3(*ctx.Inward(job, d, 0u));
        }
        if (job.Op == MeshTopologyOp::InsetRegion || job.Op == MeshTopologyOp::InsetIndividual) position += width * job.Param0 + depth * job.Param1;
        ctx.DstVertices(job)[d].Position = position;
        if (job.DstInsetBasis.Slot != InvalidSlot) {
            reinterpret_cast<device InsetVertexBasis *>(BindlessBufferMutable(uint,bindless.Buffer,job.DstInsetBasis.Slot) + job.DstInsetBasis.Offset)[d] =
                {ctx.DstVertexDomain(job).Handle(d),packed_float3(base),packed_float3(width),packed_float3(depth)};
        }
    }
    if (job.DstFaceCount == 0u) {
        const auto src = ctx.SrcVertexNormals(job);
        float3 normal = TopoBilinear(float3(src[v.x]), float3(src[v.y]), float3(src[v.z]), float3(src[v.w]), s, t);
        if (s != 0.f || t != 0.f) normal = NormalizeOrZero(normal);
        normal = TopoCopyNormal(ctx, job, v.x, d, normal);
        BindlessBufferMutable(packed_float3, bindless.Buffer, pc.Destination.BaseVertexNormalSlot)[ctx.DstVertexDomain(job).Handle(d)] = packed_float3(normal);
    }
    if (job.VertexAttributes & MeshAttributeBit_Color0) {
        ctx.SetDstVertexColor(job, d, TopoBilinear(ctx.SrcVertexColor(job, v.x), ctx.SrcVertexColor(job, v.y), ctx.SrcVertexColor(job, v.z), ctx.SrcVertexColor(job, v.w), s, t));
    }
    // Skin weights and point/line material indices take the nearest source.
    const uint nearest = t < 0.5f ? (s < 0.5f ? v.x : v.y) : (s < 0.5f ? v.w : v.z);
    if (job.HasVertexPrimitives&2u) {
        const auto from=pc.Source.VertexPrimitives,to=pc.Destination.VertexPrimitives;
        const uint primitive=(job.HasVertexPrimitives&1u) ? BindlessBuffer(uint,bindless.ElementPrimitiveBuffer,from.ValuesSlot)[ElementAttributeIndex(bindless,from,ctx.SrcVertexDomain(job).Handle(nearest))] : 0u;
        BindlessBufferMutable(uint,bindless.ElementPrimitiveBuffer,to.ValuesSlot)[ElementAttributeIndex(bindless,to,ctx.DstVertexDomain(job).Handle(d))]=
            job.PrimitiveWork.Storage.Slot!=InvalidSlot ? WorkRank(bindless,job.PrimitiveWork,primitive) : primitive;
    }
    if (job.HasSkin) {
        const auto skin=ctx.SrcBoneDeform(job,nearest);
        ctx.SetDstSkin(job,d,skin);
    }
    if (job.MorphTargetCount) {
        for (uint k = 0u; k < job.MorphTargetCount; ++k) {
            const MorphTargetVertex ta = ctx.SrcMorphTarget(job,v.x,k), tb = ctx.SrcMorphTarget(job,v.y,k);
            const MorphTargetVertex tc = ctx.SrcMorphTarget(job,v.z,k), td = ctx.SrcMorphTarget(job,v.w,k);
            const MorphTargetVertex value{
                .PositionDelta = TopoBilinear(float3(ta.PositionDelta), float3(tb.PositionDelta), float3(tc.PositionDelta), float3(td.PositionDelta), s, t),
                .NormalDelta = TopoBilinear(float3(ta.NormalDelta), float3(tb.NormalDelta), float3(tc.NormalDelta), float3(td.NormalDelta), s, t),
            };
            ctx.SetDstMorphTarget(job,d,k,value);
        }
    }
}

// Each canonical output corner interpolates its attribute payload once.
kernel void TopologyGatherCorners(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshTopologyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const TopoContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshTopologyJob job = ctx.Jobs()[tile.x];
    const uint h = tile.y * ScanTileSize + lane;
    if (h >= job.DstHalfedgeCount || job.CornerAttributes == 0u) return;
    // Loose edges have no face-corner attributes, including when their endpoints lie on a surface.
    if (ctx.DstHalfedgeFaces(job)[h] == InvalidOffset || ctx.CornerMap(job)[TopoCornerMapWords*h]==InvalidOffset) {
        if (job.CornerAttributes & MeshAttributeBit_Tangent) ctx.SetDstCornerTangent(job,h,float4(0));
        if (job.CornerAttributes & MeshAttributeBit_Color0) ctx.SetDstCornerColor(job,h,float4(1));
        for (uint set=0u;set<4u;++set)
            if (job.CornerAttributes & (MeshAttributeBit_TexCoord0 << set)) ctx.SetDstCornerUv(job,set,h,float2(0));
        return;
    }
    device const uint *map = ctx.CornerMap(job) + TopoCornerMapWords * h;
    uint a = map[0], b = map[1], c = map[2], d = map[3];
    if (job.Op==MeshTopologyOp::Decimate) {
        const uint root=ctx.VertexTargets(job)[ctx.SrcCorners(job)[a]];
        // Collapsible vertices have no UV/color seam. Give all incident
        // corners the same surviving sample, without changing normal provenance.
        if (ctx.FlagVertices(job)[root]&TopoListed) {
            const auto fan=ctx.SrcFan(job,root);
            if (fan.y) a=b=c=d=ctx.SrcFanCorner(job,fan.x);
        }
    }
    const float s = as_type<float>(map[4]), t = as_type<float>(map[5]);
    if (job.CornerAttributes & MeshAttributeBit_Tangent) {
        ctx.SetDstCornerTangent(job, h, TopoBilinear(ctx.SrcCornerTangent(job, a), ctx.SrcCornerTangent(job, b), ctx.SrcCornerTangent(job, c), ctx.SrcCornerTangent(job, d), s, t));
    }
    if (job.CornerAttributes & MeshAttributeBit_Color0) {
        ctx.SetDstCornerColor(job, h, TopoBilinear(ctx.SrcCornerColor(job, a), ctx.SrcCornerColor(job, b), ctx.SrcCornerColor(job, c), ctx.SrcCornerColor(job, d), s, t));
    }
    for (uint set = 0u; set < 4u; ++set)
        if (job.CornerAttributes & (MeshAttributeBit_TexCoord0 << set)) {
            ctx.SetDstCornerUv(job, set, h, TopoBilinear(ctx.SrcCornerUv(job, set, a), ctx.SrcCornerUv(job, set, b), ctx.SrcCornerUv(job, set, c), ctx.SrcCornerUv(job, set, d), s, t));
        }
}

// Transfer directions between canonical polygon frames after output derivation.
// A new corner interpolates actual normals, never polar/azimuth angles.
inline float3 TopoDerivedCornerNormal(TopoContext ctx, constant MeshTopologyArenas &arenas, ConnectivityView conn, uint h) {
    const uint face = conn.HalfedgeFace(h);
    if (BindlessBuffer(uchar,ctx.B.Buffer, arenas.FaceSharpnessSlot)[face] != 0u) {
        return float3(BindlessBuffer(packed_float3,ctx.B.Buffer, arenas.BaseFaceNormalSlot)[face]);
    }
    const uint root = CornerSectorRoot(ctx.B, arenas.CornerSectors, h);
    if (root != InvalidOffset) {
        return float3(BindlessBuffer(NormalSector,ctx.B.Buffer, arenas.NormalSectors.ValuesSlot)[ElementAttributeIndex(ctx.B, arenas.NormalSectors, root)].Normal);
    }
    const uint v = BindlessBuffer(uint,ctx.B.IndexBuffer, arenas.CornerSlot)[h];
    return float3(BindlessBuffer(packed_float3,ctx.B.Buffer, arenas.BaseVertexNormalSlot)[v]);
}

inline CornerNormalFrame TopoCornerFrame(TopoContext ctx, constant MeshTopologyArenas &arenas, ConnectivityView conn, uint h, float3 normal) {
    const auto position = [&](uint corner) {
        const uint v = BindlessBuffer(uint,ctx.B.IndexBuffer, arenas.CornerSlot)[corner];
        return float3(BindlessBuffer(Vertex,ctx.B.VertexBuffer, arenas.VertexSlot)[v].Position);
    };
    return ComputeCornerFrame(normal, position(h), position(conn.Next(h)), position(conn.Previous(h)));
}

kernel void TopologyCustomNormals(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshTopologyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const TopoContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshTopologyJob job = ctx.Jobs()[tile.x];
    const uint d = tile.y * ScanTileSize + lane;
    if (!(job.CornerAttributes & MeshAttributeBit_Normal)) return;
    if (d < job.RetainedNormalCornerCount) {
        const uint h = WorkGroupElement(bindless,job.RetainedNormalCorners,d);
        if (ctx.Src(job).HalfedgeFace(h) != InvalidOffset && WorkRank(bindless,job.SrcHalfedgeWork,h) == InvalidOffset) {
            const float2 offset = CustomNormalOffset(bindless,pc.Source.CustomNormals,h);
            if (offset.x >= 0.f) {
                const float3 before = TopoDerivedCornerNormal(ctx,pc.Source,ctx.Src(job),h);
                const float3 normal = DecodeNormalOffset(offset,TopoCornerFrame(ctx,pc.Source,ctx.Src(job),h,before));
                const float3 after = TopoDerivedCornerNormal(ctx,pc.Destination,ctx.Dst(job),h);
                const float2 result = EncodeNormalOffset(normal,TopoCornerFrame(ctx,pc.Destination,ctx.Dst(job),h,after));
                BindlessBufferMutable(packed_float2,bindless.Buffer,pc.Destination.CustomNormals.ValuesSlot)[ElementAttributeIndex(bindless,pc.Destination.CustomNormals,h)] = result;
            }
        }
    }
    if (d >= job.DstHalfedgeCount) return;
    if (ctx.DstHalfedgeFaces(job)[d] == InvalidOffset) {
        const uint h=job.DstCornerOffset+d;
        BindlessBufferMutable(packed_float2,bindless.Buffer,pc.Destination.CustomNormals.ValuesSlot)[ElementAttributeIndex(bindless,pc.Destination.CustomNormals,h)] = float2(-1.f,0.f);
        return;
    }
    device const uint *map = ctx.CornerMap(job) + TopoCornerMapWords * d;
    const float s = as_type<float>(map[4]), t = as_type<float>(map[5]);
    const float4 weights = float4((1.f - s) * (1.f - t), s * (1.f - t), s * t, (1.f - s) * t);
    const auto src = ctx.Src(job);
    const auto dst = ctx.Dst(job);
    float3 normal = float3(0);
    bool authored = false;
    for (uint k = 0u; k < 4u; ++k) {
        if (weights[k] == 0.f) continue;
        const uint h = map[k];
        if (h==InvalidOffset || src.HalfedgeFace(h)==InvalidOffset) continue;
        const float2 offset = CustomNormalOffset(bindless, pc.Source.CustomNormals, h);
        float3 n = TopoDerivedCornerNormal(ctx, pc.Source, src, h);
        if (offset.x >= 0.f) {
            authored = true;
            n = DecodeNormalOffset(offset, TopoCornerFrame(ctx, pc.Source, src, h, n));
        }
        normal += weights[k] * n;
    }
    const uint hd = job.DstCornerOffset + d;
    float2 result = float2(-1.f, 0.f);
    if (authored && dot(normal, normal) > 1e-20f) {
        normal = TopoCopyNormal(ctx, job, ctx.SrcCorners(job)[map[0]], ctx.CornerProvenance(job)[d].x, normal);
        const uint fd = ctx.CornerProvenance(job)[d].y;
        const uint f = ctx.FaceMap(job)[fd];
        if (f != InvalidOffset && ctx.SrcSelectedFace(job, f)) {
            if (job.Op == MeshTopologyOp::FlipNormals) normal = -normal;
            else if (job.Op==MeshTopologyOp::ExtrudeRegion) {
                const uint entry=ctx.FaceEntry(job,f);
                const uint first=ctx.Counts(job,TopoCountFaces)[entry], end=ctx.Counts(job,TopoCountFaces)[entry+1u];
                if (fd>=first && fd+1u<end) normal=-normal;
            }
            else if (TopoFlipsCopies(job) && TopoRegionDuplicates(ctx, job)) {
                const uint first = ctx.Counts(job, TopoCountFaces)[ctx.FaceEntry(job, f)];
                if (fd > first && fd - first <= job.Steps) normal = -normal;
            }
        }
        normal = NormalizeOrZero(normal);
        if (dot(normal, normal) > 0.f) {
            result = EncodeNormalOffset(normal, TopoCornerFrame(ctx, pc.Destination, dst, hd, TopoDerivedCornerNormal(ctx, pc.Destination, dst, hd)));
        }
    }
    BindlessBufferMutable(packed_float2, bindless.Buffer, pc.Destination.CustomNormals.ValuesSlot)[ElementAttributeIndex(bindless, pc.Destination.CustomNormals, hd)] = result;
}


// Carries edge sharpness and edge selection through the corner map once the output connectivity numbers its edges.
kernel void TopologyEdgeAttributes(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshTopologyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const TopoContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshTopologyJob job = ctx.Jobs()[tile.x];
    const uint hd = tile.y * ScanTileSize + lane;
    if (hd >= job.DstHalfedgeCount) return;
    const auto dst = ctx.Dst(job);
    if (!dst.EdgeFirst(job.DstCornerOffset + hd)) return;
    const uint e = dst.Edge(job.DstCornerOffset + hd);
    const uint source = ctx.CornerEdgeSource(job, hd);
    ctx.DstEdgeSharpness(job)[e] = source != InvalidOffset ? ctx.SrcEdgeSharpness(job)[ctx.SrcEdge(job, source)] : uchar(0);
    const bool hidden=job.EditorState && source!=InvalidOffset && ctx.Selected({pc.Source.EdgeHiddenSlot,0u},ctx.SrcEdgeDomain(job).Handle(ctx.SrcEdge(job,source)));
    if (job.EditorState) ctx.Select({pc.Destination.EdgeHiddenSlot,0u},e,hidden);
    ctx.SelectDstEdge(job, e, !hidden && ctx.CornerSelected(job, hd));
}

#endif
