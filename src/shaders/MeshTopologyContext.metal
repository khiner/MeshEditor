#ifndef MESHTOPOLOGYCONTEXT_MSL
#define MESHTOPOLOGYCONTEXT_MSL

// The topology operators' view of a job: source and output arenas, scratch runs, selection, and the shared per-operator rules.
#include "Bindless.metal"
#include "BlockScan.metal"
#include "ConnectivityRead.metal"
#include "gpu/MeshTopologyJob.h"
#include "gpu/MeshTopologyOp.h"
#include "gpu/MeshTopologyPushConstants.h"
#include "gpu/FanItemEncoding.h"

constant uint TopoTagged = 1u;
constant uint TopoKept = 2u;
constant uint TopoInRegion = 4u; // A vertex of a selected face
constant uint TopoNeedsCopy = 8u; // A vertex on an edge between a selected and an unselected face, or of a selected edge
constant uint TopoOnBoundary = 16u; // A vertex on a selected face's open edge
constant uint TopoDissolvable = 32u; // A vertex a dissolve may drop once two edges remain at it
constant uint TopoListed = 64u; // A vertex the job's selection list names
constant uint TopoDissolved = 4u; // A halfedge whose edge a dissolve removes
constant uint TopoSide = 1u; // A halfedge that extrudes a side quad
constant uint TopoSideFlip = 2u; // A side quad wound against the halfedge's own face
constant uint TopoDelOrig = 1u; // The state bit: the extruded region borders unselected faces
constant uint TopoCountVertices = 0u;
constant uint TopoCountFaces = 1u;
constant uint TopoCountCorners = 2u;
constant uint TopoVertexMapWords = 6u;
constant uint TopoCornerMapWords = 8u;

// Writes a map entry's four sources and its bilinear weights.
inline void TopoWriteMap(device uint *map, uint4 sources, float s, float t) {
    map[0] = sources.x;
    map[1] = sources.y;
    map[2] = sources.z;
    map[3] = sources.w;
    map[4] = as_type<uint>(s);
    map[5] = as_type<uint>(t);
}

struct TopoContext {
    device const BindlessSet &B;
    constant MeshTopologyPushConstants &Pc;

    device const MeshTopologyJob *Jobs() const { return BindlessBuffer(MeshTopologyJob, B.Buffer, Pc.JobsSlot); }
    device const uint2 *Tiles() const { return BindlessBuffer(uint2, B.Buffer, Pc.TileMapSlot); }
    device uint *Scratch() const { return BindlessBufferMutable(uint, B.Buffer, Pc.ScratchSlot); }
    device atomic_uint *Atomic(device uint *p) const { return reinterpret_cast<device atomic_uint *>(p); }
    uint2 Tile(uint group_id) const { return Tiles()[Pc.FirstTile + group_id]; }

    device uint *Bits() const { return BindlessBufferMutable(uint, B.Buffer, Pc.SelectionBitsSlot); }
    bool Selected(uint word_offset, uint i) const { return (Bits()[word_offset + (i >> 5u)] >> (i & 31u)) & 1u; }
    void Select(uint word_offset, uint i) const { atomic_fetch_or_explicit(&Atomic(Bits())[word_offset + (i >> 5u)], 1u << (i & 31u), memory_order_relaxed); }

    device const uint *SrcCorners(MeshTopologyJob job) const { return BindlessBuffer(uint, B.IndexBuffer, Pc.CornerSlot) + job.SrcCornerOffset; }
    device uint *DstCorners(MeshTopologyJob job) const { return BindlessBufferMutable(uint, B.IndexBuffer, Pc.CornerSlot) + job.DstCornerOffset; }
    ConnectivityView Src(MeshTopologyJob job) const {
        return {BindlessBuffer(uint, B.Buffer, Pc.ConnectivitySlot) + job.SrcConnectivityOffset, job.SrcVertexCount, job.SrcHalfedgeCount, job.SrcFaceCount, job.SrcFaceStarts != 0u};
    }
    ConnectivityView Dst(MeshTopologyJob job) const { return {DstConnectivity(job), job.DstVertexCount, job.DstHalfedgeCount, job.DstFaceCount, job.DstFaceStarts != 0u}; }
    device uint *DstConnectivity(MeshTopologyJob job) const { return BindlessBufferMutable(uint, B.Buffer, Pc.ConnectivitySlot) + job.DstConnectivityOffset; }
    device const Vertex *SrcVertices(MeshTopologyJob job) const { return BindlessBuffer(Vertex, B.VertexBuffer, Pc.VertexSlot) + job.SrcVertexOffset; }
    device Vertex *DstVertices(MeshTopologyJob job) const { return BindlessBufferMutable(Vertex, B.VertexBuffer, Pc.VertexSlot) + job.DstVertexOffset; }
    device const uint *SrcFaceFirstTriangles(MeshTopologyJob job) const { return BindlessBuffer(uint, B.ObjectIdBuffer, Pc.FaceFirstTriangleSlot) + job.SrcFaceFirstTriangleOffset; }
    device uint *DstFaceFirstTriangles(MeshTopologyJob job) const { return BindlessBufferMutable(uint, B.ObjectIdBuffer, Pc.FaceFirstTriangleSlot) + job.DstFaceFirstTriangleOffset; }
    device uint *DstTriangleFaceIds(MeshTopologyJob job) const { return BindlessBufferMutable(uint, B.ObjectIdBuffer, Pc.TriangleFaceIdSlot) + job.DstTriangleFaceIdOffset; }
    device const uchar *SrcEdgeSharpness(MeshTopologyJob job) const { return BindlessBuffer(uchar, B.Buffer, Pc.EdgeSharpnessSlot) + job.SrcEdgeSharpnessOffset; }
    device uchar *DstEdgeSharpness(MeshTopologyJob job) const { return BindlessBufferMutable(uchar, B.Buffer, Pc.EdgeSharpnessSlot) + job.DstEdgeSharpnessOffset; }
    device const uchar *SrcFaceSharpness(MeshTopologyJob job) const { return BindlessBuffer(uchar, B.Buffer, Pc.FaceSharpnessSlot) + job.SrcFaceFirstTriangleOffset; }
    device uchar *DstFaceSharpness(MeshTopologyJob job) const { return BindlessBufferMutable(uchar, B.Buffer, Pc.FaceSharpnessSlot) + job.DstFaceFirstTriangleOffset; }
    device const uint *SrcElementPrimitives(MeshTopologyJob job) const { return BindlessBuffer(uint, B.ElementPrimitiveBuffer, Pc.ElementPrimitiveSlot) + job.SrcElementPrimitiveOffset; }
    device uint *DstElementPrimitives(MeshTopologyJob job) const { return BindlessBufferMutable(uint, B.ElementPrimitiveBuffer, Pc.ElementPrimitiveSlot) + job.DstElementPrimitiveOffset; }
    device const BoneDeformVertex *SrcBoneDeform(MeshTopologyJob job) const { return BindlessBuffer(BoneDeformVertex, B.BoneDeformBuffer, Pc.BoneDeformSlot) + job.SrcBoneDeformOffset; }
    device BoneDeformVertex *DstBoneDeform(MeshTopologyJob job) const { return BindlessBufferMutable(BoneDeformVertex, B.BoneDeformBuffer, Pc.BoneDeformSlot) + job.DstBoneDeformOffset; }
    device const MorphTargetVertex *SrcMorphTargets(MeshTopologyJob job) const { return BindlessBuffer(MorphTargetVertex, B.MorphTargetBuffer, Pc.MorphTargetSlot) + job.SrcMorphTargetOffset; }
    device MorphTargetVertex *DstMorphTargets(MeshTopologyJob job) const { return BindlessBufferMutable(MorphTargetVertex, B.MorphTargetBuffer, Pc.MorphTargetSlot) + job.DstMorphTargetOffset; }
    device const packed_float4 *SrcCornerTangents(MeshTopologyJob job) const { return BindlessBuffer(packed_float4, B.CornerTangentBuffer, Pc.CornerTangentSlot) + job.SrcCornerTangentOffset; }
    device packed_float4 *DstCornerTangents(MeshTopologyJob job) const { return BindlessBufferMutable(packed_float4, B.CornerTangentBuffer, Pc.CornerTangentSlot) + job.DstCornerTangentOffset; }
    device const packed_float4 *SrcCornerColors(MeshTopologyJob job) const { return BindlessBuffer(packed_float4, B.CornerColorBuffer, Pc.CornerColorSlot) + job.SrcCornerColorOffset; }
    device packed_float4 *DstCornerColors(MeshTopologyJob job) const { return BindlessBufferMutable(packed_float4, B.CornerColorBuffer, Pc.CornerColorSlot) + job.DstCornerColorOffset; }
    device const packed_float2 *SrcCornerUvs(MeshTopologyJob job, uint set) const { return BindlessBuffer(packed_float2, B.CornerUvBuffer, Pc.CornerUvSlot) + job.SrcCornerUvOffsets[set]; }
    device packed_float2 *DstCornerUvs(MeshTopologyJob job, uint set) const { return BindlessBufferMutable(packed_float2, B.CornerUvBuffer, Pc.CornerUvSlot) + job.DstCornerUvOffsets[set]; }
    device const packed_uint2 *SrcCustomMasks(MeshTopologyJob job) const { return BindlessBuffer(packed_uint2, B.Buffer, Pc.CustomCornerMaskSlot) + job.SrcCustomCornerMaskOffset; }
    device packed_uint2 *DstCustomMasks(MeshTopologyJob job) const { return BindlessBufferMutable(packed_uint2, B.Buffer, Pc.CustomCornerMaskSlot) + job.DstCustomCornerMaskOffset; }
    device atomic_uint *DstCustomMaskWords(MeshTopologyJob job) const { return BindlessBufferMutable(atomic_uint, B.Buffer, Pc.CustomCornerMaskSlot) + 2u * job.DstCustomCornerMaskOffset; }
    device const packed_float2 *SrcCustomNormals(MeshTopologyJob job) const { return BindlessBuffer(packed_float2, B.Buffer, Pc.CustomCornerNormalSlot) + job.SrcCustomCornerNormalOffset; }
    device const packed_float3 *SrcVertexNormals(MeshTopologyJob job) const { return BindlessBuffer(packed_float3, B.Buffer, Pc.BaseVertexNormalSlot) + job.SrcVertexOffset; }
    device const packed_float3 *SrcFaceNormals(MeshTopologyJob job) const { return BindlessBuffer(packed_float3, B.Buffer, Pc.BaseFaceNormalSlot) + job.SrcFaceFirstTriangleOffset; }
    device const uint *Lists(MeshTopologyJob job) const { return BindlessBuffer(uint, B.Buffer, Pc.ListSlot) + job.ListOffset; }
    float3 SrcPosition(MeshTopologyJob job, uint v) const { return float3(SrcVertices(job)[v].Position); }
    // The source corners at a vertex, as (face | loop << shift) items of its fan table.
    device const uint *SrcFanItems(MeshTopologyJob job, uint v, thread uint &count) const {
        device const uint *fan = BindlessBuffer(uint, B.Buffer, Pc.AdjacencySlot) + job.SrcFanAdjacencyOffset;
        count = fan[v + 1u] - fan[v];
        return fan + job.SrcVertexCount + 1u + fan[v];
    }
    uint SrcFanHalfedge(MeshTopologyJob job, uint item) const {
        return SrcFaceRange(job, item & uint(FanItemEncoding::FaceMask)).x + (item >> uint(FanItemEncoding::LoopShift));
    }
    // One source corner at `v`, for corners without a source of their own.
    uint SrcAnyCornerAt(MeshTopologyJob job, uint v) const {
        uint count;
        device const uint *items = SrcFanItems(job, v, count);
        return count > 0u ? SrcFanHalfedge(job, items[0]) : 0u;
    }
    device packed_float2 *DstCustomNormals(MeshTopologyJob job) const { return BindlessBufferMutable(packed_float2, B.Buffer, Pc.CustomCornerNormalSlot) + job.DstCustomCornerNormalOffset; }

    // Source topology.
    // A staged face index makes an n-gon source's face lookups one load.
    uint SrcFaceOf(MeshTopologyJob job, uint h) const { return job.SrcFaceOffset == InvalidOffset ? h / 3u : Scratch()[job.SrcFaceOffset + h]; }
    uint SrcPrev(MeshTopologyJob job, uint h) const {
        if (job.SrcFaceOffset == InvalidOffset) return ConnectivityPrevious(h);
        const uint2 range = SrcFaceRange(job, SrcFaceOf(job, h));
        return h == range.x ? range.y - 1u : h - 1u;
    }
    uint2 SrcFaceRange(MeshTopologyJob job, uint f) const { return Src(job).FaceHalfedges(f); }
    uint SrcOpposite(MeshTopologyJob job, uint h) const { return Src(job).Opposite(h); }
    uint SrcEdge(MeshTopologyJob job, uint h) const { return Src(job).Edge(h); }
    uint SrcEdgeHalfedge(MeshTopologyJob job, uint e) const { return Src(job).EdgeHalfedge(e); }
    bool SrcEdgeFirst(MeshTopologyJob job, uint h) const { return Src(job).EdgeFirst(h); }
    // The fan-order corner slot that holds source corner `h`'s attributes.
    uint SrcFanCorner(MeshTopologyJob job, uint h) const {
        const uint f = SrcFaceOf(job, h);
        const uint k = h - SrcFaceRange(job, f).x;
        const uint first = 3u * SrcFaceFirstTriangles(job)[f];
        return k == 0u ? first : k == 1u ? first + 1u : first + 3u * (k - 2u) + 2u;
    }
    // The job's flags may select everything, or the elements its list names in place of the source bits.
    bool SrcSelectedVertex(MeshTopologyJob job, uint v) const {
        if (job.Flags & TopologyFlagSelectAll) return true;
        if (job.Flags & TopologyFlagListSelects) return (FlagVertices(job)[v] & TopoListed) != 0u;
        return Selected(job.SrcVertexBitsOffset, v);
    }
    bool SrcSelectedEdge(MeshTopologyJob job, uint e) const {
        if (job.Flags & TopologyFlagSelectAll) return true;
        if ((job.Flags & TopologyFlagListSelects) && job.Op == MeshTopologyOp::Subdivide) return EdgeParams(job)[e] != 0u;
        return Selected(job.SrcEdgeBitsOffset, e);
    }
    bool SrcSelectedFace(MeshTopologyJob job, uint f) const { return (job.Flags & TopologyFlagSelectAll) || Selected(job.SrcFaceBitsOffset, f); }

    // Scratch runs.
    device uint *FlagVertices(MeshTopologyJob job) const { return Scratch() + job.FlagVertexOffset; }
    device uint *VertexTargets(MeshTopologyJob job) const { return Scratch() + job.VertexTargetOffset; }
    device uint *State(MeshTopologyJob job) const { return Scratch() + job.StateOffset; }
    bool DelOrig(MeshTopologyJob job) const { return job.Op == MeshTopologyOp::InsetRegion || (State(job)[0] & TopoDelOrig) != 0u; }
    // The label run holds four face arrays then two vertex arrays.
    uint LabelRun(MeshTopologyJob job, uint index) const { return job.LabelOffset + min(index, 4u) * job.SrcFaceCount + (index > 4u ? job.SrcVertexCount : 0u); }
    device uint *FaceLabels(MeshTopologyJob job) const { return Scratch() + LabelRun(job, 0u); }
    device uint *RegionBoundary(MeshTopologyJob job) const { return Scratch() + LabelRun(job, 1u); }
    device uint *RegionStart(MeshTopologyJob job) const { return Scratch() + LabelRun(job, 2u); }
    device uint *WalkLength(MeshTopologyJob job) const { return Scratch() + LabelRun(job, 3u); }
    device uint *VertexEdgeTotal(MeshTopologyJob job) const { return Scratch() + LabelRun(job, 4u); }
    device uint *VertexEdgeDissolved(MeshTopologyJob job) const { return Scratch() + LabelRun(job, 5u); }
    device uint *HalfedgeAux(MeshTopologyJob job) const { return Scratch() + job.HalfedgeAuxOffset; }
    device uint *VertexOverride(MeshTopologyJob job, uint v) const { return Scratch() + job.VertexOverrideOffset + 4u * v; }
    device uint *EdgeParams(MeshTopologyJob job) const { return Scratch() + job.HalfedgeAuxOffset; }
    float3 TransformCopy(MeshTopologyJob job, float3 p) const { return job.CopyRotation.Unpack() * p + float3(job.CopyTranslation); }
    float PlaneDistance(MeshTopologyJob job, float3 p) const { return dot(float3(job.PlaneNormal), p) - job.PlaneOffset; }
    device uint *Table(MeshTopologyJob job) const { return Scratch() + job.TableOffset; }
    // One of the two inward vectors of output vertex `d`.
    device packed_float3 *Inward(MeshTopologyJob job, uint d, uint slot) const { return reinterpret_cast<device packed_float3 *>(Scratch() + job.VertexInwardOffset) + 2u * d + slot; }
    uint SrcNext(MeshTopologyJob job, uint h) const {
        const uint2 range = SrcFaceRange(job, SrcFaceOf(job, h));
        return h + 1u < range.y ? h + 1u : range.x;
    }
    device uint *FlagHalfedges(MeshTopologyJob job) const { return Scratch() + job.FlagHalfedgeOffset; }
    device uint *FlagFaces(MeshTopologyJob job) const { return Scratch() + job.FlagFaceOffset; }
    device uint *Counts(MeshTopologyJob job, uint quantity) const { return Scratch() + job.CountsOffset + quantity * job.CountEntries; }
    // Writes one count entry's vertices, faces, and corners.
    void WriteCounts(MeshTopologyJob job, uint entry, uint3 counts) const {
        Counts(job, TopoCountVertices)[entry] = counts.x;
        Counts(job, TopoCountFaces)[entry] = counts.y;
        Counts(job, TopoCountCorners)[entry] = counts.z;
    }
    uint VertexEntry(uint v) const { return v; }
    uint HalfedgeEntry(MeshTopologyJob job, uint h) const { return job.SrcVertexCount + h; }
    uint FaceEntry(MeshTopologyJob job, uint f) const { return job.SrcVertexCount + job.SrcHalfedgeCount + f; }
    device uint *VertexMap(MeshTopologyJob job) const { return Scratch() + job.VertexMapOffset; }
    device uint *CornerMap(MeshTopologyJob job) const { return Scratch() + job.CornerMapOffset; }
    device uint *FaceMap(MeshTopologyJob job) const { return Scratch() + job.FaceMapOffset; }

    // Output topology.
    device uint *DstFaceStarts(MeshTopologyJob job) const { return DstConnectivity(job) + job.DstVertexCount + 2u * job.DstHalfedgeCount; }
    uint2 DstFaceRange(MeshTopologyJob job, uint f) const { return Dst(job).FaceHalfedges(f); }
    // Output corners in fan order: three per output triangle.
    uint DstFanCornerTotal(MeshTopologyJob job) const { return 3u * (job.DstHalfedgeCount - 2u * job.DstFaceCount); }
    // The output halfedge whose corner map fan corner `i` reads.
    uint DstFanCornerHalfedge(MeshTopologyJob job, uint i) const {
        const uint tri = i / 3u, slot = i % 3u;
        const uint fd = DstTriangleFaceIds(job)[tri] - 1u;
        const uint fan = tri - DstFaceFirstTriangles(job)[fd];
        const uint k = slot == 0u ? 0u : slot == 1u ? fan + 1u : fan + 2u;
        return DstFaceRange(job, fd).x + k;
    }

    // An output element interpolates four sources bilinearly: lerp(lerp(a, b, s), lerp(d, c, s), t).
    void WriteVertexMap4(MeshTopologyJob job, uint d, uint4 sources, float s, float t) const { TopoWriteMap(VertexMap(job) + TopoVertexMapWords * d, sources, s, t); }
    void WriteVertexMap(MeshTopologyJob job, uint d, uint a, uint b, float weight) const { WriteVertexMap4(job, d, uint4(a, b, b, a), weight, 0.f); }
    uint CornerEdgeSource(MeshTopologyJob job, uint d) const { return CornerMap(job)[TopoCornerMapWords * d + 6u]; }
    bool CornerSelected(MeshTopologyJob job, uint d) const { return CornerMap(job)[TopoCornerMapWords * d + 7u] != 0u; }
    // Writes one output corner: its vertex, its attribute sources, the source edge its arriving segment keeps, and its edge selection.
    void WriteCorner(MeshTopologyJob job, uint d, uint v_out, uint a, uint b, float weight, uint edge_source, bool selected) const {
        WriteCorner4(job, d, v_out, uint4(a, b, b, a), weight, 0.f, edge_source, selected);
    }
    void WriteCorner4(MeshTopologyJob job, uint d, uint v_out, uint4 sources, float s, float t, uint edge_source, bool selected) const {
        DstCorners(job)[d] = v_out;
        device uint *map = CornerMap(job) + TopoCornerMapWords * d;
        TopoWriteMap(map, sources, s, t);
        map[6] = edge_source;
        map[7] = selected ? 1u : 0u;
    }
};

// One kernel thread: its context, its tile's job, and its index in the pass's domain.
template<typename T>
inline T TopoBilinear(T a, T b, T c, T d, float s, float t) { return mix(mix(a, b, s), mix(d, c, s), t); }

inline bool TopoTransformsCopies(MeshTopologyJob job) { return (job.Flags & TopologyFlagTransformCopies) != 0u; }
inline bool TopoFlipsCopies(MeshTopologyJob job) { return (job.Flags & TopologyFlagFlipCopies) != 0u || job.Op == MeshTopologyOp::Solidify; }

// An extruded region that borders unselected faces moves onto copied vertices. A region that borders nothing duplicates.
inline bool TopoRegionMoves(TopoContext ctx, MeshTopologyJob job) { return TopologyBaseOp(job.Op) == MeshTopologyOp::ExtrudeRegion && ctx.DelOrig(job); }
inline bool TopoRegionDuplicates(TopoContext ctx, MeshTopologyJob job) {
    return TopologyBaseOp(job.Op) == MeshTopologyOp::DuplicateFaces || (TopologyBaseOp(job.Op) == MeshTopologyOp::ExtrudeRegion && !ctx.DelOrig(job));
}

// The grid cell of a position at the merge distance, hashed for the table.
inline int3 TopoMergeCell(float3 p, float distance) { return int3(floor(p / max(distance, 1e-9f))); }
inline uint TopoCellHash(int3 cell) {
    return (uint(cell.x) * 73856093u) ^ (uint(cell.y) * 19349663u) ^ (uint(cell.z) * 83492791u);
}
inline bool TopoEdgeDissolved(TopoContext ctx, MeshTopologyJob job, uint h) { return (ctx.FlagHalfedges(job)[h] & TopoDissolved) != 0u; }

// A dissolve drops a vertex left with no edges, and a dissolvable vertex left with exactly two.
inline bool TopoVertexRemoved(TopoContext ctx, MeshTopologyJob job, uint v) {
    if (!TopologyIsDissolve(job.Op)) return false;
    const uint total = ctx.VertexEdgeTotal(job)[v], remaining = total - ctx.VertexEdgeDissolved(job)[v];
    if (total > 0u && remaining == 0u) return true;
    return remaining == 2u && (ctx.FlagVertices(job)[v] & TopoDissolvable) != 0u;
}

// The corner count of a face loop after its corners map through the vertex targets, with repeated targets and removed vertices dropped.
inline uint TopoMappedLoopLength(TopoContext ctx, MeshTopologyJob job, uint f) {
    const uint2 range = ctx.SrcFaceRange(job, f);
    if (!TopologyIsMerge(job.Op) && !TopologyIsDissolve(job.Op)) return range.y - range.x;
    device const uint *corners = ctx.SrcCorners(job);
    device const uint *targets = ctx.VertexTargets(job);
    uint count = 0u, first = InvalidOffset, previous = InvalidOffset;
    for (uint h = range.x; h < range.y; ++h) {
        const uint v = corners[h];
        if (TopoVertexRemoved(ctx, job, v)) continue;
        const uint m = targets[v];
        if (m == previous) continue;
        if (first == InvalidOffset) first = m;
        previous = m;
        ++count;
    }
    return count > 1u && previous == first ? count - 1u : count;
}

// The boundary halfedge after `h` around a dissolved region: the next in its face, crossing every dissolved edge it meets.
inline uint TopoNextBoundary(TopoContext ctx, MeshTopologyJob job, uint h) {
    uint n = ctx.SrcNext(job, h);
    for (uint i = 0u; i < 4096u && TopoEdgeDissolved(ctx, job, n); ++i) n = ctx.SrcNext(job, ctx.SrcOpposite(job, n));
    return n;
}

// Walks a region's boundary from its start halfedge and counts the corners that remain, emitting them when `base` is valid.
// Returns zero when the walk does not close in the region's boundary count.
inline uint TopoWalkRegion(TopoContext ctx, MeshTopologyJob job, uint root, uint fd, uint base) {
    const uint start = ctx.RegionStart(job)[root];
    const uint boundary = ctx.RegionBoundary(job)[root];
    if (start == InvalidOffset || boundary == 0u) return 0u;
    device const uint *corners = ctx.SrcCorners(job);
    device const uint *vertex_offsets = ctx.Counts(job, TopoCountVertices);
    uint h = start, steps = 0u, kept = 0u;
    do {
        if (++steps > boundary) return 0u;
        const uint v = corners[h];
        if (!TopoVertexRemoved(ctx, job, v)) {
            if (base != InvalidOffset) ctx.WriteCorner(job, base + kept, vertex_offsets[v], h, h, 0.f, h, ctx.SrcSelectedEdge(job, ctx.SrcEdge(job, h)));
            ++kept;
        }
        h = TopoNextBoundary(ctx, job, h);
    } while (h != start);
    return steps == boundary ? kept : 0u;
}

// Whether a dissolve emits a source face's own loop, which happens when its region root's walk failed.
inline bool TopoDissolveOwnLoop(TopoContext ctx, MeshTopologyJob job, uint f) {
    const uint root = ctx.FaceLabels(job)[f];
    return ctx.WalkLength(job)[root] == 0u;
}

// Whether the operator removes a source face's own loop.
inline bool TopoFaceDeleted(TopoContext ctx, MeshTopologyJob job, uint f) {
    const uint2 range = ctx.SrcFaceRange(job, f);
    switch (job.Op) {
        case MeshTopologyOp::DeleteVertices:
            for (uint h = range.x; h < range.y; ++h) {
                if (ctx.SrcSelectedVertex(job, ctx.SrcCorners(job)[h])) return true;
            }
            return false;
        case MeshTopologyOp::DeleteEdges:
        case MeshTopologyOp::DeleteOnlyEdgesFaces:
            for (uint h = range.x; h < range.y; ++h) {
                if (ctx.SrcSelectedEdge(job, ctx.SrcEdge(job, h))) return true;
            }
            return false;
        case MeshTopologyOp::DeleteFaces:
        case MeshTopologyOp::DeleteOnlyFaces:
            if (job.Flags & TopologyFlagPlaneSide) {
                float3 center = float3(0.f);
                for (uint h = range.x; h < range.y; ++h) center += ctx.SrcPosition(job, ctx.SrcCorners(job)[h]);
                return ctx.PlaneDistance(job, center / float(range.y - range.x)) < 0.f;
            }
            return ctx.SrcSelectedFace(job, f);
        case MeshTopologyOp::ExtrudeFacesIndividual:
        case MeshTopologyOp::InsetIndividual:
            return ctx.SrcSelectedFace(job, f);
        case MeshTopologyOp::KeepSelectedFaces:
            return !ctx.SrcSelectedFace(job, f);
        case MeshTopologyOp::MergeAtTarget:
        case MeshTopologyOp::MergeByDistance:
        case MeshTopologyOp::MergeCollapse:
        case MeshTopologyOp::DissolveDegenerate:
            return TopoMappedLoopLength(ctx, job, f) < 3u;
        default:
            return false;
    }
}

inline bool TopoVertexKept(TopoContext ctx, MeshTopologyJob job, uint v) {
    const uint flags = ctx.FlagVertices(job)[v];
    switch (job.Op) {
        case MeshTopologyOp::DeleteVertices: return !ctx.SrcSelectedVertex(job, v);
        case MeshTopologyOp::DeleteEdges:
        case MeshTopologyOp::DeleteFaces: return (flags & TopoTagged) == 0u || (flags & TopoKept) != 0u;
        case MeshTopologyOp::DeleteLoose:
        case MeshTopologyOp::KeepSelectedFaces: return (flags & TopoKept) != 0u;
        case MeshTopologyOp::BevelEdges:
        case MeshTopologyOp::BevelVertices: return (flags & TopoInRegion) == 0u;
        case MeshTopologyOp::MergeAtTarget:
        case MeshTopologyOp::MergeByDistance:
        case MeshTopologyOp::MergeCollapse:
        case MeshTopologyOp::DissolveDegenerate: return ctx.VertexTargets(job)[v] == v;
        case MeshTopologyOp::DissolveVertices:
        case MeshTopologyOp::DissolveEdges:
        case MeshTopologyOp::DissolveFaces:
        case MeshTopologyOp::DissolveLimited: return !TopoVertexRemoved(ctx, job, v);
        default: return true;
    }
}

// How many copies a source vertex gains, placed right after its own output.
inline uint TopoVertexCopies(TopoContext ctx, MeshTopologyJob job, uint v) {
    const uint flags = ctx.FlagVertices(job)[v];
    switch (TopologyBaseOp(job.Op)) {
        case MeshTopologyOp::ExtrudeRegion:
            if (!(flags & TopoInRegion)) return 0u;
            return !ctx.DelOrig(job) || (flags & (TopoNeedsCopy | TopoOnBoundary)) ? job.Steps : 0u;
        case MeshTopologyOp::DuplicateFaces: return (flags & TopoInRegion) ? 1u : 0u;
        case MeshTopologyOp::SplitFaces:
        case MeshTopologyOp::ExtrudeEdges: return (flags & TopoNeedsCopy) ? 1u : 0u;
        default: return 0u;
    }
}

// Whether a selected face's corner at `v` moves onto the vertex's copy.
inline bool TopoFaceUsesCopy(TopoContext ctx, MeshTopologyJob job, bool face_selected, uint v) {
    if (!face_selected) return false;
    if (job.Op != MeshTopologyOp::SplitFaces && !TopoRegionMoves(ctx, job)) return false;
    return TopoVertexCopies(ctx, job, v) != 0u;
}

inline bool TopoHalfedgeMakesSide(TopoContext ctx, MeshTopologyJob job, uint h) {
    if (!(ctx.FlagHalfedges(job)[h] & TopoSide)) return false;
    return job.Op == MeshTopologyOp::Solidify || TopologyBaseOp(job.Op) != MeshTopologyOp::ExtrudeRegion || ctx.DelOrig(job);
}

inline bool TopoOriginalVertexSelected(TopoContext ctx, MeshTopologyJob job, uint v) {
    const uint flags = ctx.FlagVertices(job)[v];
    switch (TopologyBaseOp(job.Op)) {
        case MeshTopologyOp::ExtrudeRegion: return ctx.DelOrig(job) && (flags & TopoInRegion) && TopoVertexCopies(ctx, job, v) == 0u;
        case MeshTopologyOp::DuplicateFaces:
        case MeshTopologyOp::SplitFaces:
        case MeshTopologyOp::ExtrudeEdges:
        case MeshTopologyOp::ExtrudeFacesIndividual: return false;
        case MeshTopologyOp::KeepSelectedFaces: return true;
        default: return ctx.SrcSelectedVertex(job, v);
    }
}

// Writes one output face's loop bookkeeping: its start, source, and selection.
inline void TopoEmitFace(TopoContext ctx, MeshTopologyJob job, uint fd, uint base, uint source, bool selected) {
    ctx.FaceMap(job)[fd] = source;
    if (job.DstFaceStarts != 0u) ctx.DstFaceStarts(job)[fd] = base;
    if (selected) ctx.Select(job.DstFaceBitsOffset, fd);
}

#endif
