#ifndef MESHTOPOLOGY_MSL
#define MESHTOPOLOGY_MSL

// Runs one edit-mode topology operator from a source mesh and its selection into a new mesh.
// Every operator marks, counts, scans, scatters, and gathers.
// Source elements produce output vertices, faces, and corners at scanned offsets, and index maps carry every attribute domain into the output.
#include "MeshTopologyBevel.metal"
#include "MeshTopologyContext.metal"
#include "MeshTopologyFaces.metal"
#include "MeshTopologySubdivide.metal"

// The vertices, faces, and corners a source face produces.
inline uint3 TopoFaceOutputs(TopoContext ctx, MeshTopologyJob job, uint f) {
    const uint2 range = ctx.SrcFaceRange(job, f);
    const uint n = range.y - range.x;
    const bool selected = ctx.SrcSelectedFace(job, f);
    switch (TopologyBaseOp(job.Op)) {
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
        case MeshTopologyOp::AddFaces:
            return uint3(0u, 1u, n);
        case MeshTopologyOp::ExtrudeRegion:
        case MeshTopologyOp::DuplicateFaces:
            return selected && TopoRegionDuplicates(ctx, job) ? uint3(0u, 1u + job.Steps, n * (1u + job.Steps)) : uint3(0u, 1u, n);
        case MeshTopologyOp::ExtrudeFacesIndividual:
            return selected ? uint3(n, 1u + n, 5u * n) : uint3(0u, 1u, n);
        case MeshTopologyOp::Subdivide:
        case MeshTopologyOp::ConnectVertices: {
            SubdivideEmitter emitter{ctx, job, f, false, 0u, 0u, 0u, 0u};
            const uint interior = TopoSplitFace(ctx, job, f, emitter);
            return uint3(interior, emitter.Faces, emitter.Corners);
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

// Stages each halfedge's face for a source whose faces are not all triangles.
kernel void TopologyFaceIndex(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshTopologyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const TopoContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshTopologyJob job = ctx.Jobs()[tile.x];
    const uint f = tile.y * ScanTileSize + lane;
    if (job.SrcFaceOffset == InvalidOffset || f >= job.SrcFaceCount) return;
    const uint2 range = ctx.SrcFaceRange(job, f);
    device uint *faces = ctx.Scratch() + job.SrcFaceOffset;
    for (uint h = range.x; h < range.y; ++h) faces[h] = f;
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
    ctx.FlagVertices(job)[v] = job.Op == MeshTopologyOp::DissolveVertices && ctx.SrcSelectedVertex(job, v) ? TopoDissolvable : 0u;
    ctx.VertexTargets(job)[v] = job.Op == MeshTopologyOp::MergeAtTarget && ctx.SrcSelectedVertex(job, v) ? job.TargetVertex : v;
    if (job.Op == MeshTopologyOp::MergeCollapse) ctx.VertexOverride(job, v)[0] = 0u;
    if (TopologyIsDissolve(job.Op)) {
        ctx.VertexEdgeTotal(job)[v] = 0u;
        ctx.VertexEdgeDissolved(job)[v] = 0u;
    }
    if (v == 0u) ctx.State(job)[0] = 0u;
}

// Flags the halfedge's edge when the dissolve removes it, counts its ends' edges, and marks the ends the dissolve may drop.
inline void TopoMarkDissolvedEdge(TopoContext ctx, MeshTopologyJob job, uint h) {
    device const uint *corners = ctx.SrcCorners(job);
    const uint to = corners[h], from = corners[ctx.SrcPrev(job, h)];
    const uint opposite = ctx.SrcOpposite(job, h);
    bool dissolved = opposite != InvalidOffset;
    if (dissolved) {
        switch (job.Op) {
            case MeshTopologyOp::DissolveVertices: dissolved = ctx.SrcSelectedVertex(job, to) || ctx.SrcSelectedVertex(job, from); break;
            case MeshTopologyOp::DissolveEdges: dissolved = ctx.SrcSelectedEdge(job, ctx.SrcEdge(job, h)); break;
            case MeshTopologyOp::DissolveLimited: {
                const uint f = ctx.SrcFaceOf(job, h), g = ctx.SrcFaceOf(job, opposite);
                dissolved = ctx.SrcSelectedFace(job, f) && ctx.SrcSelectedFace(job, g) &&
                    dot(normalize(float3(ctx.SrcFaceNormals(job)[f])), normalize(float3(ctx.SrcFaceNormals(job)[g]))) >= cos(job.Param0);
                break;
            }
            default: dissolved = ctx.SrcSelectedFace(job, ctx.SrcFaceOf(job, h)) && ctx.SrcSelectedFace(job, ctx.SrcFaceOf(job, opposite)); break;
        }
    }
    if (dissolved) ctx.FlagHalfedges(job)[h] |= TopoDissolved;
    if (!ctx.SrcEdgeFirst(job, h)) return;
    device atomic_uint *totals = ctx.Atomic(ctx.VertexEdgeTotal(job));
    atomic_fetch_add_explicit(&totals[to], 1u, memory_order_relaxed);
    atomic_fetch_add_explicit(&totals[from], 1u, memory_order_relaxed);
    if (!dissolved) return;
    device atomic_uint *removed = ctx.Atomic(ctx.VertexEdgeDissolved(job));
    atomic_fetch_add_explicit(&removed[to], 1u, memory_order_relaxed);
    atomic_fetch_add_explicit(&removed[from], 1u, memory_order_relaxed);
    if (job.Op != MeshTopologyOp::DissolveEdges || (job.Flags & TopologyFlagKeepVertices)) return;
    device atomic_uint *flags = ctx.Atomic(ctx.FlagVertices(job));
    atomic_fetch_or_explicit(&flags[to], TopoDissolvable, memory_order_relaxed);
    atomic_fetch_or_explicit(&flags[from], TopoDissolvable, memory_order_relaxed);
}

// Collapses an edge shorter than the distance toward its lower vertex.
inline void TopoMarkShortEdge(TopoContext ctx, MeshTopologyJob job, uint h) {
    if (!ctx.SrcEdgeFirst(job, h)) return;
    device const uint *corners = ctx.SrcCorners(job);
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
    const uint h = tile.y * ScanTileSize + lane;
    if (h >= job.SrcHalfedgeCount) return;
    device atomic_uint *flags = ctx.Atomic(ctx.FlagVertices(job));
    device const uint *corners = ctx.SrcCorners(job);
    const uint to = corners[h], from = corners[ctx.SrcPrev(job, h)];
    const auto mark_ends = [&](uint bits) {
        atomic_fetch_or_explicit(&flags[to], bits, memory_order_relaxed);
        atomic_fetch_or_explicit(&flags[from], bits, memory_order_relaxed);
    };
    uint halfedge_flags = 0u;
    switch (TopologyBaseOp(job.Op)) {
        case MeshTopologyOp::DeleteEdges:
            if (ctx.SrcEdgeFirst(job, h) && ctx.SrcSelectedEdge(job, ctx.SrcEdge(job, h))) mark_ends(TopoTagged);
            break;
        case MeshTopologyOp::ExtrudeRegion:
        case MeshTopologyOp::SplitFaces: {
            if (!ctx.SrcSelectedFace(job, ctx.SrcFaceOf(job, h))) break;
            const uint opposite = ctx.SrcOpposite(job, h);
            const bool neighbor_selected = opposite != InvalidOffset && ctx.SrcSelectedFace(job, ctx.SrcFaceOf(job, opposite));
            if (neighbor_selected) break;
            const bool region = TopologyBaseOp(job.Op) == MeshTopologyOp::ExtrudeRegion;
            if (region) halfedge_flags = TopoSide;
            if (opposite == InvalidOffset) {
                if (region) mark_ends(TopoOnBoundary);
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
    if (TopologyIsDissolve(job.Op)) TopoMarkDissolvedEdge(ctx, job, h);
    else if (job.Op == MeshTopologyOp::DissolveDegenerate) TopoMarkShortEdge(ctx, job, h);
    else if (job.Op == MeshTopologyOp::EdgeSplit) ctx.HalfedgeAux(job)[h] = TopoSectorRep(ctx, job, h);
    else if (job.Op == MeshTopologyOp::Subdivide && (job.Flags & (TopologyFlagListCuts | TopologyFlagScreenCuts)) && h < job.SrcEdgeCount) ctx.EdgeParams(job)[h] = InvalidOffset;
    else if (job.Op == MeshTopologyOp::Subdivide && (job.Flags & TopologyFlagListSelects) && h < job.SrcEdgeCount) ctx.EdgeParams(job)[h] = 0u;
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
    device const uint *corners = ctx.SrcCorners(job);
    const uint mark = (deleted ? TopoTagged : TopoKept) | (ctx.SrcSelectedFace(job, f) ? TopoInRegion : 0u);
    for (uint h = range.x; h < range.y; ++h) atomic_fetch_or_explicit(&flags[corners[h]], mark, memory_order_relaxed);
    // A bevel removes every vertex it moves corners off, marked as in the region.
    if (TopologyIsBevel(job.Op)) {
        for (uint h = range.x; h < range.y; ++h) {
            if (TopoVertexBeveled(ctx, job, corners[h])) atomic_fetch_or_explicit(&flags[corners[h]], TopoInRegion, memory_order_relaxed);
        }
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
    const uint h = tile.y * ScanTileSize + lane;
    const bool dissolve = TopologyIsDissolve(job.Op);
    if (h >= job.SrcHalfedgeCount || !(dissolve || job.Op == MeshTopologyOp::MergeCollapse)) return;
    if (dissolve ? !TopoEdgeDissolved(ctx, job, h) : !ctx.SrcEdgeFirst(job, h)) return;
    device const uint *corners = ctx.SrcCorners(job);
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
    const uint h = tile.y * ScanTileSize + lane;
    if (!TopologyIsDissolve(job.Op) || h >= job.SrcHalfedgeCount || !TopoEdgeDissolved(ctx, job, h)) return;
    if (ctx.WalkLength(job)[ctx.FaceLabels(job)[ctx.SrcFaceOf(job, h)]] != 0u) return;
    ctx.FlagHalfedges(job)[h] &= ~TopoDissolved;
    if (!ctx.SrcEdgeFirst(job, h)) return;
    device const uint *corners = ctx.SrcCorners(job);
    device atomic_uint *removed = ctx.Atomic(ctx.VertexEdgeDissolved(job));
    atomic_fetch_sub_explicit(&removed[corners[h]], 1u, memory_order_relaxed);
    atomic_fetch_sub_explicit(&removed[corners[ctx.SrcPrev(job, h)]], 1u, memory_order_relaxed);
}

kernel void TopologyMergeTable(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshTopologyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const TopoContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshTopologyJob job = ctx.Jobs()[tile.x];
    const uint i = tile.y * ScanTileSize + lane;
    if (job.Op != MeshTopologyOp::MergeByDistance || i > job.TableMask) return;
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

// A limited dissolve drops vertices left with two edges that nearly continue each other.
kernel void TopologyDissolveLimitVertices(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshTopologyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const TopoContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshTopologyJob job = ctx.Jobs()[tile.x];
    const uint v = tile.y * ScanTileSize + lane;
    if (job.Op != MeshTopologyOp::DissolveLimited || v >= job.SrcVertexCount || !ctx.SrcSelectedVertex(job, v)) return;
    if (ctx.VertexEdgeTotal(job)[v] - ctx.VertexEdgeDissolved(job)[v] != 2u) return;
    // The two remaining edges are the undissolved edges at the vertex's corners.
    uint count;
    device const uint *items = ctx.SrcFanItems(job, v, count);
    device const uint *corners = ctx.SrcCorners(job);
    float3 directions[2];
    uint found = 0u;
    for (uint i = 0u; i < count && found < 2u; ++i) {
        const uint h = ctx.SrcFanHalfedge(job, items[i]);
        const uint out = ctx.SrcNext(job, h);
        if (!TopoEdgeDissolved(ctx, job, h)) {
            const float3 d = normalize(ctx.SrcPosition(job, corners[ctx.SrcPrev(job, h)]) - ctx.SrcPosition(job, v));
            if (found == 0u || dot(directions[0], d) < 0.999f) directions[found++] = d;
        }
        if (found < 2u && !TopoEdgeDissolved(ctx, job, out)) {
            const float3 d = normalize(ctx.SrcPosition(job, corners[out]) - ctx.SrcPosition(job, v));
            if (found == 0u || dot(directions[0], d) < 0.999f) directions[found++] = d;
        }
    }
    if (found < 2u) return;
    // Collinear edges point opposite ways, so the angle between them measures the bend.
    if (dot(directions[0], directions[1]) <= -cos(job.Param0)) ctx.FlagVertices(job)[v] |= TopoDissolvable;
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
    const uint i = tile.y * ScanTileSize + lane;
    if (job.Flags & TopologyFlagScreenCuts) {
        if (i >= job.SrcEdgeCount) return;
        device const uint *corners = ctx.SrcCorners(job);
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
    if (job.Flags & TopologyFlagListSelects) {
        const uint element = list[1u + i];
        if (job.Op == MeshTopologyOp::Subdivide) {
            if (element < job.SrcEdgeCount) ctx.EdgeParams(job)[element] = 1u;
        } else if (element < job.SrcVertexCount) {
            atomic_fetch_or_explicit(&ctx.Atomic(ctx.FlagVertices(job))[element], TopoListed, memory_order_relaxed);
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
        if (opposite == InvalidOffset || ctx.FaceLabels(job)[ctx.SrcFaceOf(job, opposite)] != InvalidOffset) continue;
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
    if (ctx.WalkLength(job)[g] != opposite) return;
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
    if (!TopologyDisplaces(job.Op, job.Flags) || d >= job.DstVertexCount) return;
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
    const uint own = job.Op == MeshTopologyOp::EdgeSplit ? TopoSectorCount(ctx, job, v) : (TopoVertexKept(ctx, job, v) ? 1u : 0u);
    uint faces = 0u, corners = 0u;
    if (TopologyIsBevel(job.Op) && (ctx.FlagVertices(job)[v] & TopoInRegion)) {
        // The vertex's polygon indexes the halfedge outputs' offsets, so the ring is counted here and emitted after the scan.
        uint ring[BevelMaxRing], ring_source[BevelMaxRing];
        corners = TopoBevelRing(ctx, job, v, ring, ring_source);
        faces = corners >= 3u ? 1u : 0u;
        corners = faces * corners;
    }
    ctx.WriteCounts(job, entry, uint3(own + TopoVertexCopies(ctx, job, v), faces, corners));
}

kernel void TopologyCountHalfedges(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshTopologyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const TopoContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshTopologyJob job = ctx.Jobs()[tile.x];
    const uint h = tile.y * ScanTileSize + lane;
    if (h >= job.SrcHalfedgeCount) return;
    const uint entry = ctx.HalfedgeEntry(job, h);
    const bool side = TopoHalfedgeMakesSide(ctx, job, h);
    const bool cut = job.Op == MeshTopologyOp::Subdivide && ctx.SrcEdgeFirst(job, h) && TopoEdgeCut(ctx, job, h);
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
    if (f == job.SrcFaceCount && job.Op == MeshTopologyOp::AddFaces && job.ListOffset != InvalidOffset) {
        device const uint *list = ctx.Lists(job);
        listed.x = list[0];
        uint cursor = 1u + 3u * listed.x;
        listed.y = list[cursor++];
        for (uint i = 0u; i < listed.y; ++i) {
            listed.z += list[cursor];
            cursor += 1u + list[cursor];
        }
    }
    ctx.WriteCounts(job, entry, listed);
}

// One in-place scan the pass parameter selects: its counts, block sums, and whether the job runs it.
struct TopoScan {
    device uint *Counts;
    uint Entries;
    device uint *Blocks;
    uint BlockCount;
    bool Active;
};

inline TopoScan TopoScanOf(TopoContext ctx, MeshTopologyJob job, uint quantity) {
    switch (TopologyScan(ctx.Pc.PassParameter)) {
        case ScanCounts:
            return {ctx.Counts(job, quantity), job.CountEntries, ctx.Scratch() + job.CountBlockOffset + quantity * job.CountBlockCount, job.CountBlockCount, true};
        default:
            return {ctx.Scratch() + job.CustomPopcountOffset, job.CustomWordCount + 1u, ctx.Scratch() + job.CustomBlockOffset, job.CustomBlockCount, job.DstCustomCornerMaskOffset != InvalidOffset};
    }
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
    if (!scan.Active) return;
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
    // The count scan runs its three quantities quantity-major over the count blocks.
    const uint quantities = TopologyScan(pc.PassParameter) == ScanCounts ? 3u : 1u;
    for (uint quantity = 0u; quantity < quantities; ++quantity) {
        const TopoScan scan = TopoScanOf(ctx, job, quantity);
        if (!scan.Active) return;
        ScanBlockPrefix(scan.Blocks, scan.BlockCount, lane, simd_lane, simd_group, sums);
    }
}

// Each thread overwrites only its own counts with their exclusive offsets, permitting an in-place scan.
// The custom normal scan writes each mask word's rank into the mask and keeps the total as its terminator.
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
    if (!scan.Active) return;
    uint local[ScanPerThread];
    uint start = ScanBlockStart(scan.Counts, scan.Entries, block, scan.Blocks, lane, simd_lane, simd_group, sums, local);
    const bool masks = TopologyScan(pc.PassParameter) == ScanCustomNormals;
    const uint base = block * ScanBlockElements + lane * ScanPerThread;
    for (uint k = 0u; k < ScanPerThread; ++k) {
        const uint i = base + k;
        if (i >= scan.Entries) break;
        if (masks && i < job.CustomWordCount) ctx.DstCustomMasks(job)[i].y = start;
        else scan.Counts[i] = start;
        start += local[k];
    }
}

// Clears the output selection masks and custom-normal mask words before the scatter sets bits.
kernel void TopologyZeroOutput(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshTopologyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const TopoContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshTopologyJob job = ctx.Jobs()[tile.x];
    const uint w = tile.y * ScanTileSize + lane;
    device uint *bits = BindlessBufferMutable(uint, bindless.Buffer, pc.SelectionBitsSlot);
    if (w < ConnectivityWordCount(job.DstVertexCount)) bits[job.DstVertexBitsOffset + w] = 0u;
    if (w < ConnectivityWordCount(job.DstHalfedgeCount)) bits[job.DstEdgeBitsOffset + w] = 0u;
    if (w < ConnectivityWordCount(job.DstFaceCount)) bits[job.DstFaceBitsOffset + w] = 0u;
    if (job.DstCustomCornerMaskOffset != InvalidOffset && w < job.CustomWordCount) ctx.DstCustomMasks(job)[w] = packed_uint2(0u, 0u);
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
        // A beveled vertex becomes the polygon of the points around it, wound against the strips' rows.
        device const uint *face_offsets = ctx.Counts(job, TopoCountFaces);
        if (face_offsets[entry + 1u] == face_offsets[entry]) return;
        uint ring[BevelMaxRing], ring_source[BevelMaxRing];
        const uint length = TopoBevelRing(ctx, job, v, ring, ring_source);
        const uint fd = face_offsets[entry], base = ctx.Counts(job, TopoCountCorners)[entry];
        for (uint k = 0u; k < length; ++k) {
            const uint i = length - 1u - k;
            ctx.WriteCorner(job, base + k, TopoBevelVertexOf(ctx, job, ring[i]), ring_source[i], ring_source[i], 0.f, InvalidOffset, true);
        }
        TopoEmitFace(ctx, job, fd, base, ctx.SrcFaceOf(job, ring_source[0]), true);
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
        if (TopoOriginalVertexSelected(ctx, job, v) && !(rip && count > 1u)) ctx.Select(job.DstVertexBitsOffset, d);
    }
    for (uint copy = first_copy; copy < d + count; ++copy) {
        ctx.WriteVertexMap(job, copy, v, v, 0.f);
        if (!rip || copy == d + 1u) ctx.Select(job.DstVertexBitsOffset, copy);
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
    const uint h = tile.y * ScanTileSize + lane;
    if (h >= job.SrcHalfedgeCount) return;
    const uint entry = ctx.HalfedgeEntry(job, h);
    if (TopologyIsBevel(job.Op)) {
        const BevelHalfedgeOutputs o = TopoBevelOutputs(ctx, job, h);
        device const uint *corners = ctx.SrcCorners(job);
        const uint v = corners[h], u = corners[ctx.SrcPrev(job, h)];
        const uint base = ctx.Counts(job, TopoCountVertices)[entry];
        // Every new vertex sits on an original one, displaced to its point.
        const auto place = [&](uint d, uint anchor, float3 p) {
            ctx.WriteVertexMap(job, d, anchor, anchor, 0.f);
            *ctx.Inward(job, d, 0u) = packed_float3(p - ctx.SrcPosition(job, anchor));
            ctx.Select(job.DstVertexBitsOffset, d);
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
            TopoEmitFace(ctx, job, fd, corner_base, ctx.SrcFaceOf(job, h), true);
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
        device const uint *corners = ctx.SrcCorners(job);
        const uint from = corners[ctx.SrcPrev(job, h)], to = corners[h];
        for (uint i = 0u; i < count; ++i) {
            ctx.WriteVertexMap(job, first + i, from, to, TopoCutParam(ctx, job, h, i, count));
            ctx.Select(job.DstVertexBitsOffset, first + i);
        }
        return;
    }
    device const uint *face_offsets = ctx.Counts(job, TopoCountFaces);
    if (face_offsets[entry + 1u] == face_offsets[entry]) return;
    const uint fd = face_offsets[entry];
    const uint base = ctx.Counts(job, TopoCountCorners)[entry];
    device const uint *vertex_offsets = ctx.Counts(job, TopoCountVertices);
    device const uint *corners = ctx.SrcCorners(job);
    const uint prev = ctx.SrcPrev(job, h);
    const uint a = corners[prev], b = corners[h];
    // The quad winds to match the face left beside it.
    const bool flip = (ctx.FlagHalfedges(job)[h] & TopoSideFlip) != 0u || job.Op == MeshTopologyOp::Solidify;
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
        TopoEmitFace(ctx, job, fd + layer - 1u, quad, ctx.SrcFaceOf(job, h), false);
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
    if (ctx.Counts(job, TopoCountFaces)[entry + 1u] == ctx.Counts(job, TopoCountFaces)[entry]) return;
    uint fd = ctx.Counts(job, TopoCountFaces)[entry];
    uint base = ctx.Counts(job, TopoCountCorners)[entry];
    const uint new_vertices = ctx.Counts(job, TopoCountVertices)[entry];
    const bool list_entry = f == job.SrcFaceCount;
    const uint2 range = list_entry ? uint2(0u, 0u) : ctx.SrcFaceRange(job, f);
    const uint n = range.y - range.x;
    const bool selected = !list_entry && ctx.SrcSelectedFace(job, f);
    device const uint *vertex_offsets = ctx.Counts(job, TopoCountVertices);
    device const uint *src_corners = ctx.SrcCorners(job);
    device const uint *targets = ctx.VertexTargets(job);
    if (job.Op == MeshTopologyOp::Subdivide || job.Op == MeshTopologyOp::ConnectVertices) {
        SubdivideEmitter emitter{ctx, job, f, true, fd, base, 0u, 0u};
        TopoSplitFace(ctx, job, f, emitter);
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
        TopoEmitFace(ctx, job, fd, base, f, selected);
        return;
    }
    if (job.Op == MeshTopologyOp::Triangulate && selected && n > 3u) {
        uint3 triangles[FacesMaxCorners];
        const uint count = TopoTriangulateFace(ctx, job, f, triangles);
        for (uint t = 0u; t < count; ++t) {
            const uint3 tri = triangles[t];
            const uint corner_index[3] = {tri.x, tri.y, tri.z};
            for (uint q = 0u; q < 3u; ++q) {
                const uint k = corner_index[q], from = corner_index[(q + 2u) % 3u];
                const uint h = range.x + k;
                // A triangle side along the face loop keeps that edge, and a diagonal is new.
                const bool along = (from + 1u) % n == k;
                ctx.WriteCorner(job, base + q, vertex_offsets[src_corners[h]], h, h, 0.f, along ? h : InvalidOffset, true);
            }
            TopoEmitFace(ctx, job, fd + t, base, f, true);
            base += 3u;
        }
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
        TopoEmitFace(ctx, job, fd, base, f, true);
        return;
    }
    if (job.Op == MeshTopologyOp::Poke && selected) {
        // The center vertex averages the face's corners and lifts by the offset along the face normal.
        const uint center = new_vertices;
        const uint h0 = range.x, h1 = range.x + (n >= 4u ? n / 4u : 1u), h2 = range.x + (n >= 4u ? n / 2u : 2u), h3 = range.x + (n >= 4u ? (3u * n) / 4u : 2u);
        const float t = n >= 4u ? 0.5f : 1.f / 3.f;
        ctx.WriteVertexMap4(job, center, uint4(src_corners[h0], src_corners[h1], src_corners[h2], src_corners[h3]), 0.5f, t);
        *ctx.Inward(job, center, 0u) = packed_float3(normalize(float3(ctx.SrcFaceNormals(job)[f])) * job.Param0);
        ctx.Select(job.DstVertexBitsOffset, center);
        for (uint k = 0u; k < n; ++k) {
            const uint h = range.x + k, next = range.x + (k + 1u) % n;
            ctx.WriteCorner(job, base, vertex_offsets[src_corners[h]], h, h, 0.f, InvalidOffset, true);
            ctx.WriteCorner(job, base + 1u, vertex_offsets[src_corners[next]], next, next, 0.f, next, ctx.SrcSelectedEdge(job, ctx.SrcEdge(job, next)));
            ctx.WriteCorner4(job, base + 2u, center, uint4(h0, h1, h2, h3), 0.5f, t, InvalidOffset, true);
            TopoEmitFace(ctx, job, fd + k, base, f, true);
            base += 3u;
        }
        return;
    }
    if (job.Op == MeshTopologyOp::FlipNormals && selected) {
        // The reversed loop's segment arriving at corner k comes down the edge that arrived at k + 1.
        for (uint j = 0u; j < n; ++j) {
            const uint k = n - 1u - j;
            const uint h = range.x + k, edge = range.x + (k + 1u) % n;
            ctx.WriteCorner(job, base + j, vertex_offsets[src_corners[h]], h, h, 0.f, edge, ctx.SrcSelectedEdge(job, ctx.SrcEdge(job, edge)));
        }
        TopoEmitFace(ctx, job, fd, base, f, true);
        return;
    }
    if (job.Op == MeshTopologyOp::EdgeSplit) {
        for (uint k = 0u; k < n; ++k) {
            const uint h = range.x + k;
            ctx.WriteCorner(job, base + k, TopoSectorVertex(ctx, job, h), h, h, 0.f, h, ctx.SrcSelectedEdge(job, ctx.SrcEdge(job, h)));
        }
        TopoEmitFace(ctx, job, fd, base, f, selected);
        return;
    }
    if (TopologyIsDissolve(job.Op) && !TopoDissolveOwnLoop(ctx, job, f)) {
        // A region root emits its walked boundary as one face, selected when the face was selected or the region spans more faces.
        TopoWalkRegion(ctx, job, f, fd, base);
        TopoEmitFace(ctx, job, fd, base, f, selected || ctx.RegionBoundary(job)[f] != n);
        return;
    }
    const bool individual = TopologyBaseOp(job.Op) == MeshTopologyOp::ExtrudeFacesIndividual && selected;
    const bool duplicated = selected && TopoRegionDuplicates(ctx, job);
    // A deletion, merge, or keep leaves every surviving edge's selection alone, and an extrusion selects only the moved faces' edges.
    const bool edges_from_source = job.Op <= MeshTopologyOp::DeleteLoose || job.Op == MeshTopologyOp::KeepSelectedFaces || TopologyIsDissolve(job.Op) || TopologyIsMerge(job.Op);
    const bool own_selected = individual || duplicated ? false : (job.Op == MeshTopologyOp::KeepSelectedFaces || selected) && job.Op != MeshTopologyOp::ExtrudeEdges;
    // The list entry appends the job's listed faces, whose corners inherit attributes from any corner at their vertex.
    if (f == job.SrcFaceCount) {
        if (job.Op != MeshTopologyOp::AddFaces || job.ListOffset == InvalidOffset) return;
        device const uint *list = ctx.Lists(job);
        // Listed vertices sit on source vertex zero, displaced to their listed positions.
        const uint listed = list[0];
        for (uint i = 0u; i < listed; ++i) {
            const float3 p = float3(as_type<float>(list[1u + 3u * i]), as_type<float>(list[2u + 3u * i]), as_type<float>(list[3u + 3u * i]));
            ctx.WriteVertexMap(job, new_vertices + i, 0u, 0u, 0.f);
            *ctx.Inward(job, new_vertices + i, 0u) = packed_float3(p - ctx.SrcPosition(job, 0u));
            ctx.Select(job.DstVertexBitsOffset, new_vertices + i);
        }
        uint cursor = 1u + 3u * listed;
        const uint faces = list[cursor++];
        for (uint i = 0u; i < faces; ++i) {
            const uint length = list[cursor++];
            // A listed vertex's corner inherits from the first source vertex in its face.
            uint any = 0u;
            for (uint k = 0u; k < length; ++k) {
                if (list[cursor + k] < job.SrcVertexCount) {
                    any = ctx.SrcAnyCornerAt(job, list[cursor + k]);
                    break;
                }
            }
            for (uint k = 0u; k < length; ++k) {
                const uint v = list[cursor + k];
                const uint source_corner = v < job.SrcVertexCount ? ctx.SrcAnyCornerAt(job, v) : any;
                ctx.WriteCorner(job, base + k, v < job.SrcVertexCount ? vertex_offsets[v] : new_vertices + (v - job.SrcVertexCount), source_corner, source_corner, 0.f, InvalidOffset, true);
            }
            TopoEmitFace(ctx, job, fd, base, InvalidOffset, true);
            ++fd;
            base += length;
            cursor += length;
        }
        return;
    }
    if (individual) {
        // The face's own copies of its corner vertices carry the top face, and each edge gets a quad down to the original vertices.
        for (uint k = 0u; k < n; ++k) {
            const uint h = range.x + k;
            const uint copy = new_vertices + k;
            ctx.WriteVertexMap(job, copy, src_corners[h], src_corners[h], 0.f);
            ctx.Select(job.DstVertexBitsOffset, copy);
            if (job.Op == MeshTopologyOp::InsetIndividual) {
                const uint prev = range.x + (k + n - 1u) % n, next = range.x + (k + 1u) % n;
                const float3 displacement = TopoInsetCorner(ctx.SrcPosition(job, src_corners[prev]), ctx.SrcPosition(job, src_corners[h]), ctx.SrcPosition(job, src_corners[next]), normalize(float3(ctx.SrcFaceNormals(job)[f])), job.Param0, job.Param1, (job.Flags & TopologyFlagEvenOffset) != 0u);
                *ctx.Inward(job, copy, 0u) = packed_float3(displacement);
            }
            ctx.WriteCorner(job, base + k, copy, h, h, 0.f, h, true);
        }
        TopoEmitFace(ctx, job, fd, base, f, true);
        ++fd;
        base += n;
        for (uint k = 0u; k < n; ++k) {
            const uint h = range.x + k, prev = k == 0u ? range.y - 1u : h - 1u;
            const uint a = vertex_offsets[src_corners[prev]], b = vertex_offsets[src_corners[h]];
            const uint4 loop = uint4(a, b, new_vertices + k, new_vertices + (k == 0u ? n - 1u : k - 1u));
            const uint4 sources = uint4(prev, h, h, prev);
            for (uint j = 0u; j < 4u; ++j) ctx.WriteCorner(job, base + j, loop[j], sources[j], sources[j], 0.f, j == 1u || j == 3u ? h : InvalidOffset, j == 3u);
            TopoEmitFace(ctx, job, fd, base, f, false);
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
        ctx.WriteCorner(job, base + emitted++, v_out, h, h, 0.f, h, (edges_from_source || own_selected) && ctx.SrcSelectedEdge(job, ctx.SrcEdge(job, h)));
    }
    TopoEmitFace(ctx, job, fd, base, f, own_selected && selected);
    if (!duplicated) return;
    // Each layer's duplicate loop uses that layer's copies and is selected, reversed when the copies flip.
    const bool flipped = TopoFlipsCopies(job);
    for (uint layer = 1u; layer <= job.Steps; ++layer) {
        fd += 1u;
        base += n;
        for (uint j = 0u; j < n; ++j) {
            const uint k = flipped ? n - 1u - j : j;
            const uint h = range.x + k, edge = flipped ? range.x + (k + 1u) % n : h;
            ctx.WriteCorner(job, base + j, vertex_offsets[src_corners[h]] + layer, h, h, 0.f, edge, true);
        }
        TopoEmitFace(ctx, job, fd, base, f, true);
    }
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
    // A fan gives a face two fewer triangles than corners, so its first triangle is its start less two per face before it.
    const uint2 range = ctx.DstFaceRange(job, fd);
    const uint first = range.x - 2u * fd, last = range.y - 2u * (fd + 1u);
    ctx.DstFaceFirstTriangles(job)[fd] = first;
    device uint *face_ids = ctx.DstTriangleFaceIds(job);
    for (uint t = first; t < last; ++t) face_ids[t] = fd + 1u;
    const uint source = ctx.FaceMap(job)[fd];
    ctx.DstElementPrimitives(job)[fd] = source != InvalidOffset ? ctx.SrcElementPrimitives(job)[source] : 0u;
    ctx.DstFaceSharpness(job)[fd] = source != InvalidOffset ? ctx.SrcFaceSharpness(job)[source] : uchar(0);
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
    device const Vertex *src = ctx.SrcVertices(job);
    const bool merged = TopologyIsMerge(job.Op) && v.x == job.TargetVertex;
    float3 position = merged ? float3(job.TargetPosition) : TopoBilinear(float3(src[v.x].Position), float3(src[v.y].Position), float3(src[v.z].Position), float3(src[v.w].Position), s, t);
    if (job.Op == MeshTopologyOp::MergeCollapse && ctx.VertexOverride(job, v.x)[0] != 0u) {
        device const uint *override = ctx.VertexOverride(job, v.x);
        position = float3(as_type<float>(override[1]), as_type<float>(override[2]), as_type<float>(override[3]));
    }
    if (job.Op == MeshTopologyOp::InsetRegion) {
        // A boundary copy sits between two inward directions, and an even offset keeps the inset width across the corner.
        const float3 in_a = float3(*ctx.Inward(job, d, 0u)), in_b = float3(*ctx.Inward(job, d, 1u));
        const bool has_a = dot(in_a, in_a) > 0.f, has_b = dot(in_b, in_b) > 0.f;
        if (has_a || has_b) {
            const float3 sum = in_a + in_b;
            const float scale = has_a && has_b ? ((job.Flags & TopologyFlagEvenOffset) != 0u ? 1.f / max(1.f + dot(in_a, in_b), 1e-4f) : 0.5f) : 1.f;
            position += sum * (job.Param0 * scale) + float3(ctx.SrcVertexNormals(job)[v.x]) * job.Param1;
        }
    } else if (TopologyDisplaces(job.Op, job.Flags)) {
        position += float3(*ctx.Inward(job, d, 0u));
    }
    ctx.DstVertices(job)[d].Position = position;
    // Skin weights take the nearest source.
    const uint nearest = t < 0.5f ? (s < 0.5f ? v.x : v.y) : (s < 0.5f ? v.w : v.z);
    if (job.SrcBoneDeformOffset != InvalidOffset) ctx.DstBoneDeform(job)[d] = ctx.SrcBoneDeform(job)[nearest];
    if (job.SrcMorphTargetOffset != InvalidOffset) {
        device const MorphTargetVertex *targets = ctx.SrcMorphTargets(job);
        device MorphTargetVertex *out = ctx.DstMorphTargets(job);
        for (uint k = 0u; k < job.MorphTargetCount; ++k) {
            const uint base = k * job.SrcVertexCount;
            const MorphTargetVertex ta = targets[base + v.x], tb = targets[base + v.y], tc = targets[base + v.z], td = targets[base + v.w];
            out[k * job.DstVertexCount + d] = {
                .PositionDelta = TopoBilinear(float3(ta.PositionDelta), float3(tb.PositionDelta), float3(tc.PositionDelta), float3(td.PositionDelta), s, t),
                .NormalDelta = TopoBilinear(float3(ta.NormalDelta), float3(tb.NormalDelta), float3(tc.NormalDelta), float3(td.NormalDelta), s, t),
            };
        }
    }
}

// Carries the corner layers through the corner map, one thread per output fan corner, and marks custom normals present.
kernel void TopologyGatherCorners(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshTopologyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const TopoContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshTopologyJob job = ctx.Jobs()[tile.x];
    const uint i = tile.y * ScanTileSize + lane;
    if (i >= ctx.DstFanCornerTotal(job)) return;
    const uint hd = ctx.DstFanCornerHalfedge(job, i);
    device const uint *map = ctx.CornerMap(job) + TopoCornerMapWords * hd;
    const uint a = ctx.SrcFanCorner(job, map[0]), b = ctx.SrcFanCorner(job, map[1]), c = ctx.SrcFanCorner(job, map[2]), d = ctx.SrcFanCorner(job, map[3]);
    const float s = as_type<float>(map[4]), t = as_type<float>(map[5]);
    if (job.SrcCornerTangentOffset != InvalidOffset) {
        device const packed_float4 *src = ctx.SrcCornerTangents(job);
        ctx.DstCornerTangents(job)[i] = TopoBilinear(float4(src[a]), float4(src[b]), float4(src[c]), float4(src[d]), s, t);
    }
    if (job.SrcCornerColorOffset != InvalidOffset) {
        device const packed_float4 *src = ctx.SrcCornerColors(job);
        ctx.DstCornerColors(job)[i] = TopoBilinear(float4(src[a]), float4(src[b]), float4(src[c]), float4(src[d]), s, t);
    }
    for (uint set = 0u; set < 4u; ++set) {
        if (job.SrcCornerUvOffsets[set] == InvalidOffset) continue;
        device const packed_float2 *src = ctx.SrcCornerUvs(job, set);
        ctx.DstCornerUvs(job, set)[i] = TopoBilinear(float2(src[a]), float2(src[b]), float2(src[c]), float2(src[d]), s, t);
    }
    // An exact copy keeps its authored corner normal, and an interpolated corner derives its normal.
    if (job.SrcCustomCornerMaskOffset == InvalidOffset || a != b || s != 0.f || t != 0.f) return;
    if ((ctx.SrcCustomMasks(job)[a >> 5u].x >> (a & 31u)) & 1u) {
        atomic_fetch_or_explicit(&ctx.DstCustomMaskWords(job)[2u * (i >> 5u)], 1u << (i & 31u), memory_order_relaxed);
    }
}

kernel void TopologyCustomPopcount(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshTopologyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const TopoContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshTopologyJob job = ctx.Jobs()[tile.x];
    const uint w = tile.y * ScanTileSize + lane;
    if (job.DstCustomCornerMaskOffset == InvalidOffset || w > job.CustomWordCount) return;
    ctx.Scratch()[job.CustomPopcountOffset + w] = w < job.CustomWordCount ? popcount(ctx.DstCustomMasks(job)[w].x) : 0u;
}

// Packs each present custom normal to its ranked slot, copied from the source corner's slot.
kernel void TopologyCustomPack(
    uint lane [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshTopologyPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const TopoContext ctx{bindless, pc};
    const uint2 tile = ctx.Tile(group_id);
    const MeshTopologyJob job = ctx.Jobs()[tile.x];
    const uint i = tile.y * ScanTileSize + lane;
    if (job.DstCustomCornerMaskOffset == InvalidOffset || i >= ctx.DstFanCornerTotal(job)) return;
    const packed_uint2 mask = ctx.DstCustomMasks(job)[i >> 5u];
    if (((mask.x >> (i & 31u)) & 1u) == 0u) return;
    const uint hd = ctx.DstFanCornerHalfedge(job, i);
    const uint a = ctx.SrcFanCorner(job, ctx.CornerMap(job)[TopoCornerMapWords * hd]);
    const packed_uint2 src_mask = ctx.SrcCustomMasks(job)[a >> 5u];
    const uint src_rank = src_mask.y + popcount(src_mask.x & ((1u << (a & 31u)) - 1u));
    const uint dst_rank = mask.y + popcount(mask.x & ((1u << (i & 31u)) - 1u));
    ctx.DstCustomNormals(job)[dst_rank] = ctx.SrcCustomNormals(job)[src_rank];
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
    if (!dst.EdgeFirst(hd)) return;
    const uint e = dst.Edge(hd);
    const uint source = ctx.CornerEdgeSource(job, hd);
    ctx.DstEdgeSharpness(job)[e] = source != InvalidOffset ? ctx.SrcEdgeSharpness(job)[ctx.SrcEdge(job, source)] : uchar(0);
    if (ctx.CornerSelected(job, hd)) ctx.Select(job.DstEdgeBitsOffset, e);
}

#endif
