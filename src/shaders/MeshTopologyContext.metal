#ifndef MESHTOPOLOGYCONTEXT_MSL
#define MESHTOPOLOGYCONTEXT_MSL

// The topology operators' view of a job: source and output arenas, scratch runs, selection, and the shared per-operator rules.
#include "Bindless.metal"
#include "ElementWorkShared.metal"
#include "CornerNormalOffset.metal"
#include "BlockScan.metal"
#include "ConnectivityRead.metal"
#include "gpu/MeshTopologyJob.h"
#include "gpu/MeshTopologyOp.h"
#include "gpu/MeshTopologyPushConstants.h"

constant uint TopoTagged = 1u;
constant uint TopoKept = 2u;
constant uint TopoInRegion = 4u; // A vertex in the geometry selected for this operation
constant uint TopoNeedsCopy = 8u; // A vertex the operation must copy, including a split vertex with unselected users
constant uint TopoRegionEdge = 32u; // A region vertex used by a selected edge
constant uint TopoRegionSides = 2u; // State bit: a selected edge creates side faces
constant uint TopoRegionUnselected = 0x80000000u; // Edge classification: at least one unselected face user
constant uint TopoOnBoundary = 16u; // A vertex on a selected face's open edge
constant uint TopoDissolvable = 32u; // A vertex a dissolve may drop once two edges remain at it
constant uint TopoListed = 64u; // A vertex the job's selection list names
constant uint TopoLimitedQueued = 256u; // Per-chain work queue membership
constant uint TopoLabelsConverged = 4u; // State bit: all component labels converged
constant uint TopoSurfaceVertex = 128u; // A dissolve source vertex has at least one face user
constant uint TopoDissolved = 4u; // A halfedge whose edge a dissolve removes
constant uint TopoWireRemoved = 8u; // A loose edge attached to a dissolved surface vertex
constant uint TopoSurfaceEdge = 16u; // A mapped halfedge belongs to a surviving simple face loop
constant uint TopoSide = 1u; // A halfedge that extrudes a side quad
constant uint TopoSideFlip = 2u; // A side quad wound against the halfedge's own face
constant uint TopoDelOrig = 1u; // The state bit: the extruded region borders unselected faces
constant uint TopoCountVertices = 0u;
constant uint TopoCountFaces = 1u;
constant uint TopoCountCorners = 2u; // Face corners only; explicit wire pairs follow their total.
constant uint TopoCountWireCorners = 3u;
constant uint TopoVertexMapWords = 6u;
constant uint TopoCornerMapWords = 8u;

template<typename T> struct TopoSourceValues {
    device const T *Values;
    ElementWorkDomain Domain;
    T operator[](uint index) const { return Values[Domain.Handle(index)]; }
};
struct TopoCornerVertices {
    device const uint *Handles;
    ElementWorkDomain Vertices;
    uint operator[](uint halfedge) const { return Vertices.Index(Handles[halfedge]); }
};
struct TopoHalfedgeScratch {
    device uint *Values;
    ElementWorkDomain Halfedges;
    device uint &operator[](uint handle) const { return Values[Halfedges.Index(handle)]; }
};

// Output work numbering is independent of ownership and allocation order.
// A handle map may mix retained vertices with newly reserved slots.
struct TopoOutputDomain {
    device const BindlessSet &B;
    SlotOffset Handles;
    uint Handle(uint index) const {
        return BindlessBuffer(uint, B.Buffer, Handles.Slot)[Handles.Offset + index];
    }
};
template<typename T> struct TopoOutputValues {
    device T *Values;
    TopoOutputDomain Domain;
    device T &operator[](uint index) const { return Values[Domain.Handle(index)]; }
};

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

    device uint *Storage() const { return BindlessBufferMutable(uint, B.Buffer, Pc.StorageSlot); }
    device const MeshTopologyJob *Jobs() const { return reinterpret_cast<device const MeshTopologyJob *>(Storage() + Pc.JobsOffset); }
    device const uint2 *Tiles() const { return reinterpret_cast<device const uint2 *>(Storage() + Pc.TileMapOffset); }
    device uint *Scratch() const { return Storage() + Pc.ScratchOffset; }
    device atomic_uint *Atomic(device uint *p) const { return reinterpret_cast<device atomic_uint *>(p); }
    uint2 Tile(uint group_id) const { return Tiles()[Pc.FirstTile + group_id]; }

    device uint *Bits(SlotOffset range) const { return BindlessBufferMutable(uint, B.Buffer, range.Slot) + range.Offset; }
    bool Selected(SlotOffset range, uint i) const { return (BindlessBuffer(uint,B.Buffer,range.Slot)[range.Offset+(i >> 5u)] >> (i & 31u)) & 1u; }
    void Select(SlotOffset range, uint i, bool selected = true) const {
        device atomic_uint *word = &Atomic(Bits(range))[i >> 5u];
        const uint mask = 1u << (i & 31u);
        if (selected) atomic_fetch_or_explicit(word, mask, memory_order_relaxed);
        else atomic_fetch_and_explicit(word, ~mask, memory_order_relaxed);
    }

    ElementWorkDomain SrcVertexDomain(MeshTopologyJob job) const { return {B, job.SrcVertexWork, 0u}; }
    ElementWorkDomain SrcHalfedgeDomain(MeshTopologyJob job) const { return {B, job.SrcHalfedgeWork, 0u}; }
    ElementWorkDomain SrcFaceDomain(MeshTopologyJob job) const { return {B, job.SrcFaceWork, 0u}; }
    ElementWorkDomain SrcEdgeDomain(MeshTopologyJob job) const { return {B, job.SrcEdgeWork, 0u}; }
    TopoCornerVertices SrcCorners(MeshTopologyJob job) const { return {BindlessBuffer(uint,B.IndexBuffer, Pc.Source.CornerSlot), SrcVertexDomain(job)}; }
    TopoOutputDomain DstVertexDomain(MeshTopologyJob job) const { return {B, job.DstVertexHandles}; }
    TopoOutputDomain DstFaceDomain(MeshTopologyJob job) const { return {B, job.DstFaceHandles}; }
    void SelectDstVertex(MeshTopologyJob job, uint v, bool selected = true) const { Select({job.DstVertexBits.Slot, 0u}, DstVertexDomain(job).Handle(v), selected); }
    void SelectDstFace(MeshTopologyJob job, uint f, bool selected) const { Select({job.DstFaceBits.Slot, 0u}, DstFaceDomain(job).Handle(f), selected); }
    void SelectDstEdge(MeshTopologyJob job, uint e, bool selected) const { Select({job.DstEdgeBits.Slot, 0u}, e, selected); }
    device uint *DstCorners(MeshTopologyJob job) const { return BindlessBufferMutable(uint, B.IndexBuffer, Pc.Destination.CornerSlot) + job.DstCornerOffset; }
    ConnectivityView Src(MeshTopologyJob job) const {
        return {B, job.SrcConnectivity, job.SrcFaceCount};
    }
    ConnectivityView Dst(MeshTopologyJob job) const { return {B, job.DstConnectivity, job.DstFaceCount}; }
    TopoSourceValues<Vertex> SrcVertices(MeshTopologyJob job) const { return {BindlessBuffer(Vertex,B.VertexBuffer, Pc.Source.VertexSlot), SrcVertexDomain(job)}; }
    TopoOutputValues<Vertex> DstVertices(MeshTopologyJob job) const { return {BindlessBufferMutable(Vertex, B.VertexBuffer, Pc.Destination.VertexSlot), DstVertexDomain(job)}; }
    TopoOutputValues<uint> DstFaceTriangles(MeshTopologyJob job) const { return {BindlessBufferMutable(uint, B.ObjectIdBuffer, Pc.Destination.FaceTriangleStartSlot), DstFaceDomain(job)}; }
    device packed_uint3 *DstTriangles(MeshTopologyJob job) const { return BindlessBufferMutable(packed_uint3, B.Buffer, Pc.Destination.TriangleSlot) + job.DstTriangleOffset; }
    TopoSourceValues<uchar> SrcEdgeSharpness(MeshTopologyJob job) const { return {BindlessBuffer(uchar,B.Buffer, Pc.Source.EdgeSharpnessSlot), SrcEdgeDomain(job)}; }
    device uchar *DstEdgeSharpness(MeshTopologyJob job) const { return BindlessBufferMutable(uchar, B.Buffer, Pc.Destination.EdgeSharpnessSlot); }
    TopoSourceValues<uchar> SrcFaceSharpness(MeshTopologyJob job) const { return {BindlessBuffer(uchar,B.Buffer, Pc.Source.FaceSharpnessSlot), SrcFaceDomain(job)}; }
    TopoOutputValues<uchar> DstFaceSharpness(MeshTopologyJob job) const { return {BindlessBufferMutable(uchar, B.Buffer, Pc.Destination.FaceSharpnessSlot), DstFaceDomain(job)}; }
    uint SrcElementPrimitive(MeshTopologyJob job, uint f) const { return BindlessBuffer(uint,B.ElementPrimitiveBuffer, Pc.Source.FacePrimitives.ValuesSlot)[ElementAttributeIndex(B, Pc.Source.FacePrimitives, SrcFaceDomain(job).Handle(f))]; }
    void SetDstElementPrimitive(MeshTopologyJob job, uint f, uint value) const { BindlessBufferMutable(uint, B.ElementPrimitiveBuffer, Pc.Destination.FacePrimitives.ValuesSlot)[ElementAttributeIndex(B, Pc.Destination.FacePrimitives, DstFaceDomain(job).Handle(f))] = value; }
    BoneDeformVertex SrcBoneDeform(MeshTopologyJob job, uint v) const {
        const uint handle=SrcVertexDomain(job).Handle(v);
        return BindlessBuffer(BoneDeformVertex,B.BoneDeformBuffer,Pc.Source.Skin.ValuesSlot)
            [ElementAttributeIndex(B,Pc.Source.Skin,handle)];
    }
    void SetDstSkin(MeshTopologyJob job,uint v,BoneDeformVertex value) const {
        const uint handle=DstVertexDomain(job).Handle(v);
        BindlessBufferMutable(BoneDeformVertex,B.BoneDeformBuffer,Pc.Destination.Skin.ValuesSlot)
            [ElementAttributeIndex(B,Pc.Destination.Skin,handle)]=value;
    }
    MorphTargetVertex SrcMorphTarget(MeshTopologyJob job,uint v,uint target) const {
        const uint handle=SrcVertexDomain(job).Handle(v);
        return BindlessBuffer(MorphTargetVertex,B.MorphTargetBuffer,Pc.Source.Morph.ValuesSlot)[ElementAttributeIndex(B,Pc.Source.Morph,handle,target)];
    }
    void SetDstMorphTarget(MeshTopologyJob job,uint v,uint target,MorphTargetVertex value) const {
        const uint handle=DstVertexDomain(job).Handle(v);
        BindlessBufferMutable(MorphTargetVertex,B.MorphTargetBuffer,Pc.Destination.Morph.ValuesSlot)[ElementAttributeIndex(B,Pc.Destination.Morph,handle,target)]=value;
    }
    float4 SrcCornerTangent(MeshTopologyJob job, uint h) const { return float4(BindlessBuffer(packed_float4,B.CornerTangentBuffer, Pc.Source.CornerTangent.ValuesSlot)[ElementAttributeIndex(B, Pc.Source.CornerTangent, h)]); }
    void SetDstCornerTangent(MeshTopologyJob job, uint h, float4 value) const { BindlessBufferMutable(packed_float4, B.CornerTangentBuffer, Pc.Destination.CornerTangent.ValuesSlot)[ElementAttributeIndex(B, Pc.Destination.CornerTangent, job.DstCornerOffset + h)] = value; }
    float4 SrcCornerColor(MeshTopologyJob job, uint h) const { return float4(BindlessBuffer(packed_float4,B.CornerColorBuffer, Pc.Source.CornerColor.ValuesSlot)[ElementAttributeIndex(B, Pc.Source.CornerColor, h)]); }
    void SetDstCornerColor(MeshTopologyJob job, uint h, float4 value) const { BindlessBufferMutable(packed_float4, B.CornerColorBuffer, Pc.Destination.CornerColor.ValuesSlot)[ElementAttributeIndex(B, Pc.Destination.CornerColor, job.DstCornerOffset + h)] = value; }
    float2 SrcCornerUv(MeshTopologyJob job, uint set, uint h) const { return float2(BindlessBuffer(packed_float2,B.CornerUvBuffer, Pc.Source.CornerUvs[set].ValuesSlot)[ElementAttributeIndex(B, Pc.Source.CornerUvs[set], h)]); }
    void SetDstCornerUv(MeshTopologyJob job, uint set, uint h, float2 value) const { BindlessBufferMutable(packed_float2, B.CornerUvBuffer, Pc.Destination.CornerUvs[set].ValuesSlot)[ElementAttributeIndex(B, Pc.Destination.CornerUvs[set], job.DstCornerOffset + h)] = value; }
    float4 SrcVertexColor(MeshTopologyJob job, uint v) const { return float4(BindlessBuffer(packed_float4,B.CornerColorBuffer, Pc.Source.VertexColor.ValuesSlot)[ElementAttributeIndex(B, Pc.Source.VertexColor, SrcVertexDomain(job).Handle(v))]); }
    void SetDstVertexColor(MeshTopologyJob job, uint v, float4 value) const { BindlessBufferMutable(packed_float4, B.CornerColorBuffer, Pc.Destination.VertexColor.ValuesSlot)[ElementAttributeIndex(B, Pc.Destination.VertexColor, DstVertexDomain(job).Handle(v))] = value; }
    TopoSourceValues<packed_float3> SrcVertexNormals(MeshTopologyJob job) const { return {BindlessBuffer(packed_float3,B.Buffer, Pc.Source.BaseVertexNormalSlot), SrcVertexDomain(job)}; }
    TopoSourceValues<packed_float3> SrcFaceNormals(MeshTopologyJob job) const { return {BindlessBuffer(packed_float3,B.Buffer, Pc.Source.BaseFaceNormalSlot), SrcFaceDomain(job)}; }
    device const uint *Lists(MeshTopologyJob job) const { return BindlessBuffer(uint, B.Buffer, Pc.ListSlot) + job.ListOffset; }
    float3 SrcPosition(MeshTopologyJob job, uint v) const { return float3(SrcVertices(job)[v].Position); }
    uint2 SrcFan(MeshTopologyJob job, uint v) const { const auto src = Src(job); return src.Incoming(SrcVertexDomain(job).Handle(v)); }
    uint SrcFanCorner(MeshTopologyJob job, uint index) const { return Src(job).FanCorner(index); }
    // One source corner at v, for an output corner without a source of its own.
    uint SrcAnyCornerAt(MeshTopologyJob job, uint v) const {
        const uint2 fan = SrcFan(job, v);
        return fan.y ? SrcFanCorner(job,fan.x) : SrcHalfedgeDomain(job).Handle(0u);
    }

    // Source topology.
    // Canonical ownership makes every face lookup one load.
    uint SrcFaceOf(MeshTopologyJob job, uint h) const { const auto src = Src(job); return h == InvalidOffset ? InvalidOffset : SrcFaceDomain(job).Index(src.HalfedgeFace(h)); }
    uint SrcPrev(MeshTopologyJob job, uint h) const { return Src(job).Previous(h); }
    uint2 SrcFaceRange(MeshTopologyJob job, uint f) const { return Src(job).FaceHalfedges(SrcFaceDomain(job).Handle(f)); }
    uint SrcOpposite(MeshTopologyJob job, uint h) const { const auto src = Src(job); return src.Opposite(h); }
    uint SrcEdge(MeshTopologyJob job, uint h) const { const auto src = Src(job); return SrcEdgeDomain(job).Index(src.Edge(h)); }
    uint SrcEdgeHalfedge(MeshTopologyJob job, uint e) const { const auto src = Src(job); return src.EdgeHalfedge(SrcEdgeDomain(job).Handle(e)); }
    bool SrcEdgeFirst(MeshTopologyJob job, uint h) const { return Src(job).EdgeFirst(h); }
    // The job's flags may select everything, or the elements its list names in place of the source bits.
    bool SrcSelectedVertex(MeshTopologyJob job, uint v) const {
        if (v >= job.SrcVertexCount) return false;
        if (job.SelectionElement==Element::Vertex) return Scratch()[job.SelectionOffset+v]!=0u;
        if (job.Flags & TopologyFlagSelectAll) return true;
        if (job.Flags & TopologyFlagListSelects) return (FlagVertices(job)[v] & TopoListed) != 0u;
        return Selected({job.SrcVertexBits.Slot, 0u}, SrcVertexDomain(job).Handle(v));
    }
    bool SrcSelectedEdge(MeshTopologyJob job, uint e) const {
        if (e >= job.SrcEdgeCount) return false;
        if (job.SelectionElement==Element::Edge || job.SelectionElement==Element::Vertex)
            return Scratch()[job.SelectionOffset+job.SrcVertexCount+e]!=0u;
        if (job.Flags & TopologyFlagSelectAll) return true;
        if ((job.Flags & TopologyFlagListSelects) && job.Op == MeshTopologyOp::Subdivide) return EdgeParams(job)[e] != 0u;
        return Selected({job.SrcEdgeBits.Slot, 0u}, SrcEdgeDomain(job).Handle(e));
    }
    bool SrcSelectedFace(MeshTopologyJob job,uint f) const {
        if (f>=job.SrcFaceCount) return false;
        if (job.SelectionElement==Element::Face || job.SelectionElement==Element::Vertex)
            return Scratch()[job.SelectionOffset+job.SrcVertexCount+job.SrcHalfedgeCount+f]!=0u;
        return (job.Flags&TopologyFlagSelectAll) || Selected({job.SrcFaceBits.Slot,0u},SrcFaceDomain(job).Handle(f));
    }

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
    TopoHalfedgeScratch HalfedgeAux(MeshTopologyJob job) const { return {Scratch() + job.HalfedgeAuxOffset, SrcHalfedgeDomain(job)}; }
    device uint *EdgeParams(MeshTopologyJob job) const { return Scratch() + job.HalfedgeAuxOffset; }
    device uint *WireEdgeMap(MeshTopologyJob job) const { return Scratch() + job.WireEdgeMapOffset; }
    float3 TransformCopy(MeshTopologyJob job, float3 p) const { return job.CopyRotation.Unpack() * p + float3(job.CopyTranslation); }
    float PlaneDistance(MeshTopologyJob job, float3 p) const { return dot(float3(job.PlaneNormal), p) - job.PlaneOffset; }
    device uint *Table(MeshTopologyJob job) const { return Scratch() + job.TableOffset; }
    // One of the two inward vectors of output vertex `d`.
    device packed_float3 *Inward(MeshTopologyJob job, uint d, uint slot) const { return reinterpret_cast<device packed_float3 *>(Scratch() + job.VertexInwardOffset) + 2u * d + slot; }
    uint SrcNext(MeshTopologyJob job, uint h) const { return Src(job).Next(h); }
    TopoHalfedgeScratch DissolvePrevious(MeshTopologyJob job) const { return {Scratch() + job.FaceLoopOffset, SrcHalfedgeDomain(job)}; }
    TopoHalfedgeScratch FlagHalfedges(MeshTopologyJob job) const { return {Scratch() + job.FlagHalfedgeOffset, SrcHalfedgeDomain(job)}; }
    device uint *FlagFaces(MeshTopologyJob job) const { return Scratch() + job.FlagFaceOffset; }
    device uint *Counts(MeshTopologyJob job, uint quantity) const { return Scratch() + job.CountsOffset + quantity * job.CountEntries; }
    // Face corners and wire corners occupy separate runs of the same output allocation.
    void WriteCounts(MeshTopologyJob job, uint entry, uint3 counts, uint wire_corners = 0u) const {
        Counts(job, TopoCountVertices)[entry] = counts.x;
        Counts(job, TopoCountFaces)[entry] = counts.y;
        Counts(job, TopoCountCorners)[entry] = counts.z;
        Counts(job, TopoCountWireCorners)[entry] = wire_corners;
    }
    uint WireCornerOffset(MeshTopologyJob job, uint entry) const {
        return Counts(job, TopoCountCorners)[job.CountEntries - 1u] + Counts(job, TopoCountWireCorners)[entry];
    }
    uint VertexEntry(uint v) const { return v; }
    uint HalfedgeEntry(MeshTopologyJob job, uint h) const { return job.SrcVertexCount + SrcHalfedgeDomain(job).Index(h); }
    uint FaceEntry(MeshTopologyJob job, uint f) const { return job.SrcVertexCount + job.SrcHalfedgeCount + f; }
    device uint *VertexMap(MeshTopologyJob job) const { return Scratch() + job.VertexMapOffset; }
    device uint *CornerMap(MeshTopologyJob job) const { return Scratch() + job.CornerMapOffset; }
    device uint *FaceMap(MeshTopologyJob job) const { return Scratch() + job.FaceMapOffset; }
    device packed_uint2 *CornerProvenance(MeshTopologyJob job) const { return reinterpret_cast<device packed_uint2 *>(Scratch() + job.CornerProvenanceOffset); }

    // Output topology.
    TopoOutputValues<packed_uint2> DstFaceRanges(MeshTopologyJob job) const { return {BindlessBufferMutable(packed_uint2, B.Buffer, job.DstConnectivity.FaceRanges.Slot), DstFaceDomain(job)}; }
    device uint *DstHalfedgeFaces(MeshTopologyJob job) const { return BindlessBufferMutable(uint, B.Buffer, job.DstConnectivity.HalfedgeFaces.Slot) + job.DstCornerOffset; }
    uint2 DstFaceRange(MeshTopologyJob job, uint f) const { return Dst(job).FaceHalfedges(DstFaceDomain(job).Handle(f)) - job.DstCornerOffset; }
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
        DstCorners(job)[d] = DstVertexDomain(job).Handle(v_out);
        if (job.CornerAttributes & MeshAttributeBit_Normal) CornerProvenance(job)[d].x = v_out;
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
    return TopologyBaseOp(job.Op) == MeshTopologyOp::DuplicateGeometry || (TopologyBaseOp(job.Op) == MeshTopologyOp::ExtrudeRegion && !ctx.DelOrig(job));
}

inline uint TopoRegionEdgeFaces(TopoContext ctx, MeshTopologyJob job, uint e) { return ctx.EdgeParams(job)[2u*e]&~TopoRegionUnselected; }
inline bool TopoRegionHasSides(TopoContext ctx, MeshTopologyJob job) { return (ctx.State(job)[0]&TopoRegionSides)!=0u; }
inline bool TopoRegionKeepsVertex(TopoContext ctx, MeshTopologyJob job, uint v) {
    const uint flags=ctx.FlagVertices(job)[v];
    return !(flags&TopoInRegion) || !ctx.DelOrig(job) || (flags&TopoNeedsCopy);
}
inline uint TopoRegionLayerVertex(TopoContext ctx, MeshTopologyJob job, uint v, uint layer) {
    const bool only_top=TopoRegionHasSides(ctx,job) && !(ctx.FlagVertices(job)[v]&TopoOnBoundary);
    return ctx.Counts(job,TopoCountVertices)[v]+uint(TopoRegionKeepsVertex(ctx,job,v))+(only_top ? 0u : layer-1u);
}
inline bool TopoRegionConnectsVertex(TopoContext ctx, MeshTopologyJob job, uint v) {
    return (ctx.FlagVertices(job)[v]&(TopoInRegion|TopoRegionEdge))==TopoInRegion;
}

// The grid cell of a position at the merge distance, hashed for the table.
inline int3 TopoMergeCell(float3 p, float distance) { return int3(floor(p / max(distance, 1e-9f))); }
inline uint TopoCellHash(int3 cell) {
    return (uint(cell.x) * 73856093u) ^ (uint(cell.y) * 19349663u) ^ (uint(cell.z) * 83492791u);
}
inline bool TopoEdgeDissolved(TopoContext ctx, MeshTopologyJob job, uint h) { return (ctx.FlagHalfedges(job)[h] & TopoDissolved) != 0u; }

// A core without faces holds only whole lines, each as the two corners at its ends, since face corners come with their faces.
inline bool TopoLineCore(MeshTopologyJob job) { return job.SrcFaceCount == 0u; }

// Surface cleanup drops newly isolated face vertices. Vertex dissolve also drops
// selected isolated points and joins selected vertices left with two edges.
inline bool TopoVertexRemoved(TopoContext ctx, MeshTopologyJob job, uint v) {
    if (!TopologyIsDissolve(job.Op)) return false;
    const uint flags=ctx.FlagVertices(job)[v];
    const uint total = ctx.VertexEdgeTotal(job)[v], remaining = total - ctx.VertexEdgeDissolved(job)[v];
    if (remaining==0u) return (flags&TopoSurfaceVertex)!=0u || (job.Op==MeshTopologyOp::DissolveVertices && (flags&TopoDissolvable)!=0u);
    return remaining == 2u && (flags & TopoDissolvable) != 0u;
}

// The corner count of a face loop after its corners map through the vertex targets, with repeated targets and removed vertices dropped.
inline uint TopoMappedLoopLength(TopoContext ctx, MeshTopologyJob job, uint f) {
    const uint2 range = ctx.SrcFaceRange(job, f);
    if (!TopologyIsMerge(job.Op) && !TopologyIsDissolve(job.Op)) return range.y - range.x;
    const auto corners = ctx.SrcCorners(job);
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
    const auto corners = ctx.SrcCorners(job);
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

// Edges removed directly by the selection. Face deletion instead drops edges with no surviving face.
inline bool TopoEdgeDeleted(TopoContext ctx, MeshTopologyJob job, uint h) {
    if (job.Op==MeshTopologyOp::DeleteVertices) {
        const auto corners=ctx.SrcCorners(job);
        return ctx.SrcSelectedVertex(job,corners[h]) || ctx.SrcSelectedVertex(job,corners[ctx.SrcPrev(job,h)]);
    }
    return TopologyDeletesEdges(job.Op) && ctx.SrcSelectedEdge(job,ctx.SrcEdge(job,h));
}

// Whether the operator removes a source face's own loop.
inline bool TopoFaceDeleted(TopoContext ctx, MeshTopologyJob job, uint f) {
    const uint2 range = ctx.SrcFaceRange(job, f);
    switch (job.Op) {
        case MeshTopologyOp::Wireframe:
            return (job.Flags & TopologyFlagWireReplace) && ctx.SrcSelectedFace(job, f);
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
        case MeshTopologyOp::Decimate:
        case MeshTopologyOp::DissolveDegenerate:
            return TopoMappedLoopLength(ctx, job, f) < 3u;
        default:
            return false;
    }
}

inline bool TopoVertexKept(TopoContext ctx, MeshTopologyJob job, uint v) {
    const uint flags = ctx.FlagVertices(job)[v];
    switch (job.Op) {
        case MeshTopologyOp::ExtrudeRegion: return TopoRegionKeepsVertex(ctx,job,v);
        case MeshTopologyOp::ReplaceFaces: return !(flags & TopoListed);
        case MeshTopologyOp::Wireframe: return !(flags & TopoInRegion) || (flags & TopoKept);
        case MeshTopologyOp::DeleteVertices: return !ctx.SrcSelectedVertex(job, v);
        case MeshTopologyOp::DeleteLoose:
        case MeshTopologyOp::DeleteEdges:
        case MeshTopologyOp::DeleteFaces: return (flags & TopoTagged) == 0u || (flags & TopoKept) != 0u;
        case MeshTopologyOp::KeepSelectedFaces: return (flags & TopoKept) != 0u || (job.SelectionElement==Element::Vertex && ctx.SrcSelectedVertex(job,v));
        case MeshTopologyOp::BevelEdges:
        case MeshTopologyOp::BevelVertices: return (flags & TopoInRegion) == 0u;
        case MeshTopologyOp::MergeAtTarget:
        case MeshTopologyOp::MergeByDistance:
        case MeshTopologyOp::MergeCollapse:
        case MeshTopologyOp::Decimate:
        case MeshTopologyOp::DissolveDegenerate: return ctx.VertexTargets(job)[v] == v;
        case MeshTopologyOp::DissolveVertices:
        case MeshTopologyOp::DissolveEdges:
        case MeshTopologyOp::RotateEdges:
        case MeshTopologyOp::DissolveFaces:
        case MeshTopologyOp::DissolveLimited: return !TopoVertexRemoved(ctx, job, v);
        default: return true;
    }
}

// How many copies a source vertex gains, placed right after its own output.
inline uint TopoVertexCopies(TopoContext ctx, MeshTopologyJob job, uint v) {
    const uint flags = ctx.FlagVertices(job)[v];
    switch (TopologyBaseOp(job.Op)) {
        case MeshTopologyOp::ExtrudeVertices: return uint(ctx.SrcSelectedVertex(job, v));
        case MeshTopologyOp::Wireframe: return (flags & TopoInRegion) ? 2u + uint((flags & TopoOnBoundary) != 0u) : 0u;
        case MeshTopologyOp::ExtrudeRegion:
            if (!(flags & TopoInRegion)) return 0u;
            if (job.Op==MeshTopologyOp::ExtrudeRegion) return TopoRegionHasSides(ctx,job) && !(flags&TopoOnBoundary) ? 1u : job.Steps;
            return !ctx.DelOrig(job) || (flags & (TopoNeedsCopy | TopoOnBoundary)) ? job.Steps : 0u;
        case MeshTopologyOp::DuplicateGeometry: return (flags & TopoInRegion) ? 1u : 0u;
        case MeshTopologyOp::SplitGeometry:
        case MeshTopologyOp::ExtrudeEdges: return (flags & TopoNeedsCopy) ? 1u : 0u;
        default: return 0u;
    }
}

// Whether a selected face's corner at `v` moves onto the vertex's copy.
inline bool TopoFaceUsesCopy(TopoContext ctx, MeshTopologyJob job, bool face_selected, uint v) {
    if (!face_selected) return false;
    if (job.Op != MeshTopologyOp::SplitGeometry && !TopoRegionMoves(ctx, job)) return false;
    return TopoVertexCopies(ctx, job, v) != 0u;
}

inline bool TopoHalfedgeMakesSide(TopoContext ctx, MeshTopologyJob job, uint h) {
    if (job.Op==MeshTopologyOp::ExtrudeRegion) {
        const uint e=ctx.SrcEdge(job,h);
        if (!ctx.SrcSelectedEdge(job,e) || TopoRegionEdgeFaces(ctx,job,e)>=2u) return false;
        const uint face_corner=ctx.EdgeParams(job)[2u*e+1u];
        return h==(face_corner==InvalidOffset ? ctx.SrcEdgeHalfedge(job,e) : face_corner);
    }
    if (!(ctx.FlagHalfedges(job)[h] & TopoSide)) return false;
    return job.Op == MeshTopologyOp::Solidify || TopologyBaseOp(job.Op) != MeshTopologyOp::ExtrudeRegion || ctx.DelOrig(job);
}

inline bool TopoOriginalVertexSelected(TopoContext ctx, MeshTopologyJob job, uint v) {
    const uint flags = ctx.FlagVertices(job)[v];
    switch (TopologyBaseOp(job.Op)) {
        case MeshTopologyOp::ExtrudeVertices: return false;
        case MeshTopologyOp::DissolveVertices:
            return (job.Flags & TopologyFlagListSelects) ? ctx.Selected({job.SrcVertexBits.Slot,0u},ctx.SrcVertexDomain(job).Handle(v)) : ctx.SrcSelectedVertex(job,v);
        case MeshTopologyOp::Wireframe: return false;
        case MeshTopologyOp::ExtrudeRegion: return ctx.DelOrig(job) && (flags & TopoInRegion) && TopoVertexCopies(ctx, job, v) == 0u;
        case MeshTopologyOp::SplitGeometry: return (flags & TopoInRegion) && !(flags & TopoNeedsCopy);
        case MeshTopologyOp::DuplicateGeometry:
        case MeshTopologyOp::ExtrudeEdges:
        case MeshTopologyOp::ExtrudeFacesIndividual: return false;
        case MeshTopologyOp::KeepSelectedFaces: return true;
        default: return ctx.SrcSelectedVertex(job, v);
    }
}

// Writes one output face's loop range, corner ownership, source, and selection.
inline void TopoEmitFace(TopoContext ctx, MeshTopologyJob job, uint fd, uint base, uint count, uint source, bool selected) {
    ctx.FaceMap(job)[fd] = source;
    ctx.DstFaceRanges(job)[fd] = packed_uint2(job.DstCornerOffset + base, job.DstCornerOffset + base + count);
    for (uint h = base; h < base + count; ++h) {
        ctx.DstHalfedgeFaces(job)[h] = ctx.DstFaceDomain(job).Handle(fd);
        if (job.CornerAttributes & MeshAttributeBit_Normal) ctx.CornerProvenance(job)[h].y = fd;
    }
    const bool hidden=source<job.SrcFaceCount && (!selected || job.Op==MeshTopologyOp::KeepSelectedFaces) && ctx.Selected({ctx.Pc.Source.FaceHiddenSlot,0u},ctx.SrcFaceDomain(job).Handle(source));
    ctx.Select({ctx.Pc.Destination.FaceHiddenSlot,0u},ctx.DstFaceDomain(job).Handle(fd),hidden);
    ctx.SelectDstFace(job, fd, selected && !hidden);
}

#endif
