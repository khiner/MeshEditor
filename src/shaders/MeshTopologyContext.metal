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

// A source domain's compact elements: the first source's, then a second source's from Split on.
struct TopoSourceDomain {
    ElementWorkDomain First, Second;
    uint Split;
    uint Handle(uint index) const { return index < Split ? First.Handle(index) : Second.Handle(index - Split); }
    uint Index(uint handle) const {
        const uint first = First.Index(handle);
        if (first != InvalidOffset) return first;
        const uint second = Second.Index(handle);
        return second == InvalidOffset ? second : Split + second;
    }
};
// A source domain's values at its compact elements, through the source's table entries.
template<typename T> struct TopoSourceValues {
    device const T *Values;
    device const uint *Table;
    TopoSourceDomain Domain;
    T operator[](uint index) const { return Values[Resolve<MeshElementBlockSize>(Table, Domain.Handle(index))]; }
};
// The compact source vertex of each source corner handle.
struct TopoCornerVertices {
    device const uint *Handles;
    device const uint *Table;
    TopoSourceDomain Vertices;
    uint operator[](uint halfedge) const { return Vertices.Index(Handles[Resolve<MeshElementBlockSize>(Table, halfedge)]); }
};
struct TopoHalfedgeScratch {
    device uint *Values;
    TopoSourceDomain Halfedges;
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
// An output domain's values at its compact elements, through the output's table entries.
template<typename T> struct TopoOutputValues {
    device T *Values;
    device const uint *Table;
    TopoOutputDomain Domain;
    device T &operator[](uint index) const { return Values[Resolve<MeshElementBlockSize>(Table, Domain.Handle(index))]; }
};
// An output domain's values at the handles of a run from First.
template<typename T> struct TopoOutputRun {
    device T *Values;
    device const uint *Table;
    uint First;
    device T &operator[](uint index) const { return Values[Resolve<MeshElementBlockSize>(Table, First + index)]; }
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
        if (range.Slot == InvalidSlot) return;
        device atomic_uint *word = &Atomic(Bits(range))[i >> 5u];
        const uint mask = 1u << (i & 31u);
        if (selected) atomic_fetch_or_explicit(word, mask, memory_order_relaxed);
        else atomic_fetch_and_explicit(word, ~mask, memory_order_relaxed);
    }

    // The table entries of an element domain of the source's or the output's tables.
    device const uint *Table(ElementTables t, uint domain) const {
        const SlotOffset table = domain == 0u ? t.Vertices : domain == 1u ? t.Halfedges : domain == 2u ? t.Faces : domain == 3u ? t.Edges : t.Triangles;
        return BindlessBuffer(uint, B.Buffer, table.Slot) + table.Offset;
    }
    device const uint *SrcTable(MeshTopologyJob job, uint domain) const { return Table(job.SrcConnectivity.Tables, domain); }
    device const uint *DstTable(MeshTopologyJob job, uint domain) const { return Table(job.DstConnectivity.Tables, domain); }
    // Arena addresses of source and output handles of the domains vertex, halfedge, face, edge and triangle.
    uint SrcAddress(MeshTopologyJob job, uint domain, uint handle) const { return Resolve<MeshElementBlockSize>(SrcTable(job, domain), handle); }
    uint DstAddress(MeshTopologyJob job, uint domain, uint handle) const { return Resolve<MeshElementBlockSize>(DstTable(job, domain), handle); }

    TopoSourceDomain SrcVertexDomain(MeshTopologyJob job) const { return {{B, job.SrcVertexWork}, {B, job.Src2Elements[0]}, job.SrcVertexCount}; }
    TopoSourceDomain SrcHalfedgeDomain(MeshTopologyJob job) const { return {{B, job.SrcHalfedgeWork}, {B, job.Src2Elements[1]}, job.SrcHalfedgeCount}; }
    TopoSourceDomain SrcFaceDomain(MeshTopologyJob job) const { return {{B, job.SrcFaceWork}, {B, job.Src2Elements[2]}, job.SrcFaceCount}; }
    TopoSourceDomain SrcEdgeDomain(MeshTopologyJob job) const { return {{B, job.SrcEdgeWork}, {B, job.Src2Elements[3]}, job.SrcEdgeCount}; }
    // Whether a compact source vertex or face is the second side's.
    bool SecondVertex(MeshTopologyJob job, uint v) const { return v - job.SrcVertexCount < job.Src2Counts[0]; }
    bool SecondFace(MeshTopologyJob job, uint f) const { return f - job.SrcFaceCount < job.Src2Counts[2]; }
    TopoCornerVertices SrcCorners(MeshTopologyJob job) const { return {BindlessBuffer(uint,B.IndexBuffer, Pc.Source.CornerSlot), SrcTable(job, 1u), SrcVertexDomain(job)}; }
    TopoOutputDomain DstVertexDomain(MeshTopologyJob job) const { return {B, job.DstVertexHandles}; }
    TopoOutputDomain DstFaceDomain(MeshTopologyJob job) const { return {B, job.DstFaceHandles}; }
    void SelectDstVertex(MeshTopologyJob job, uint v, bool selected = true) const { Select(job.DstVertexBits, DstAddress(job, 0u, DstVertexDomain(job).Handle(v)), selected); }
    void SelectDstFace(MeshTopologyJob job, uint f, bool selected) const { Select(job.DstFaceBits, DstAddress(job, 2u, DstFaceDomain(job).Handle(f)), selected); }
    void SelectDstEdge(MeshTopologyJob job, uint e, bool selected) const { Select(job.DstEdgeBits, DstAddress(job, 3u, e), selected); }
    TopoOutputRun<uint> DstCorners(MeshTopologyJob job) const { return {BindlessBufferMutable(uint, B.IndexBuffer, Pc.Destination.CornerSlot), DstTable(job, 1u), job.DstCornerOffset}; }
    ConnectivityView Src(MeshTopologyJob job) const {
        return {B, job.SrcConnectivity, job.SrcFaceCount};
    }
    ConnectivityView Dst(MeshTopologyJob job) const { return {B, job.DstConnectivity, job.DstFaceCount}; }
    TopoSourceValues<Vertex> SrcVertices(MeshTopologyJob job) const { return {BindlessBuffer(Vertex,B.VertexBuffer, Pc.Source.VertexSlot), SrcTable(job, 0u), SrcVertexDomain(job)}; }
    TopoOutputValues<Vertex> DstVertices(MeshTopologyJob job) const { return {BindlessBufferMutable(Vertex, B.VertexBuffer, Pc.Destination.VertexSlot), DstTable(job, 0u), DstVertexDomain(job)}; }
    TopoOutputValues<uint> DstFaceTriangles(MeshTopologyJob job) const { return {BindlessBufferMutable(uint, B.ObjectIdBuffer, Pc.Destination.FaceTriangleStartSlot), DstTable(job, 2u), DstFaceDomain(job)}; }
    TopoOutputRun<packed_uint3> DstTriangles(MeshTopologyJob job) const { return {BindlessBufferMutable(packed_uint3, B.Buffer, Pc.Destination.TriangleSlot), DstTable(job, 4u), job.DstTriangleOffset}; }
    TopoSourceValues<uchar> SrcEdgeSharpness(MeshTopologyJob job) const { return {BindlessBuffer(uchar,B.Buffer, Pc.Source.EdgeSharpnessSlot), SrcTable(job, 3u), SrcEdgeDomain(job)}; }
    ElementValues<uchar> DstEdgeSharpness(MeshTopologyJob job) const { return {BindlessBufferMutable(uchar, B.Buffer, Pc.Destination.EdgeSharpnessSlot), DstTable(job, 3u)}; }
    TopoSourceValues<uchar> SrcFaceSharpness(MeshTopologyJob job) const { return {BindlessBuffer(uchar,B.Buffer, Pc.Source.FaceSharpnessSlot), SrcTable(job, 2u), SrcFaceDomain(job)}; }
    TopoOutputValues<uchar> DstFaceSharpness(MeshTopologyJob job) const { return {BindlessBufferMutable(uchar, B.Buffer, Pc.Destination.FaceSharpnessSlot), DstTable(job, 2u), DstFaceDomain(job)}; }
    // Layer values of source handles, and of output handles of the job's runs.
    uint SrcElementPrimitive(MeshTopologyJob job, uint f) const { return BindlessBuffer(uint, B.ElementPrimitiveBuffer, Pc.Source.Layers.FacePrimitives.Slot)[Pc.Source.Layers.FacePrimitives.Offset + SrcFaceDomain(job).Handle(f)]; }
    void SetDstElementPrimitive(MeshTopologyJob job, uint f, uint value) const { BindlessBufferMutable(uint, B.ElementPrimitiveBuffer, Pc.Destination.Layers.FacePrimitives.Slot)[Pc.Destination.Layers.FacePrimitives.Offset + DstFaceDomain(job).Handle(f)] = value; }
    BoneDeformVertex SrcBoneDeform(MeshTopologyJob job, uint v) const { return BindlessBuffer(BoneDeformVertex, B.BoneDeformBuffer, Pc.Source.Layers.Skin.Slot)[Pc.Source.Layers.Skin.Offset + SrcVertexDomain(job).Handle(v)]; }
    void SetDstSkin(MeshTopologyJob job, uint v, BoneDeformVertex value) const { BindlessBufferMutable(BoneDeformVertex, B.BoneDeformBuffer, Pc.Destination.Layers.Skin.Slot)[Pc.Destination.Layers.Skin.Offset + DstVertexDomain(job).Handle(v)] = value; }
    MorphTargetVertex SrcMorphTarget(MeshTopologyJob job, uint v, uint target) const {
        return BindlessBuffer(MorphTargetVertex, B.MorphTargetBuffer, Pc.Source.Layers.Morph.Slot)[MorphTargetIndex(Pc.Source.Layers.Morph, SrcVertexDomain(job).Handle(v), target, job.MorphTargetCount)];
    }
    void SetDstMorphTarget(MeshTopologyJob job, uint v, uint target, MorphTargetVertex value) const {
        BindlessBufferMutable(MorphTargetVertex, B.MorphTargetBuffer, Pc.Destination.Layers.Morph.Slot)[MorphTargetIndex(Pc.Destination.Layers.Morph, DstVertexDomain(job).Handle(v), target, job.MorphTargetCount)] = value;
    }
    float4 SrcCornerTangent(MeshTopologyJob job, uint h) const { return float4(BindlessBuffer(packed_float4, B.CornerTangentBuffer, Pc.Source.Layers.CornerTangent.Slot)[Pc.Source.Layers.CornerTangent.Offset + h]); }
    void SetDstCornerTangent(MeshTopologyJob job, uint h, float4 value) const { BindlessBufferMutable(packed_float4, B.CornerTangentBuffer, Pc.Destination.Layers.CornerTangent.Slot)[Pc.Destination.Layers.CornerTangent.Offset + job.DstCornerOffset + h] = value; }
    float4 SrcCornerColor(MeshTopologyJob job, uint h) const { return float4(BindlessBuffer(packed_float4, B.CornerColorBuffer, Pc.Source.Layers.CornerColor.Slot)[Pc.Source.Layers.CornerColor.Offset + h]); }
    void SetDstCornerColor(MeshTopologyJob job, uint h, float4 value) const { BindlessBufferMutable(packed_float4, B.CornerColorBuffer, Pc.Destination.Layers.CornerColor.Slot)[Pc.Destination.Layers.CornerColor.Offset + job.DstCornerOffset + h] = value; }
    float2 SrcCornerUv(MeshTopologyJob job, uint set, uint h) const { return float2(BindlessBuffer(packed_float2, B.CornerUvBuffer, Pc.Source.Layers.CornerUvs[set].Slot)[Pc.Source.Layers.CornerUvs[set].Offset + h]); }
    void SetDstCornerUv(MeshTopologyJob job, uint set, uint h, float2 value) const { BindlessBufferMutable(packed_float2, B.CornerUvBuffer, Pc.Destination.Layers.CornerUvs[set].Slot)[Pc.Destination.Layers.CornerUvs[set].Offset + job.DstCornerOffset + h] = value; }
    float4 SrcVertexColor(MeshTopologyJob job, uint v) const { return float4(BindlessBuffer(packed_float4, B.CornerColorBuffer, Pc.Source.Layers.VertexColor.Slot)[Pc.Source.Layers.VertexColor.Offset + SrcVertexDomain(job).Handle(v)]); }
    void SetDstVertexColor(MeshTopologyJob job, uint v, float4 value) const { BindlessBufferMutable(packed_float4, B.CornerColorBuffer, Pc.Destination.Layers.VertexColor.Slot)[Pc.Destination.Layers.VertexColor.Offset + DstVertexDomain(job).Handle(v)] = value; }
    TopoSourceValues<packed_float3> SrcVertexNormals(MeshTopologyJob job) const { return {BindlessBuffer(packed_float3,B.Buffer, Pc.Source.BaseVertexNormalSlot), SrcTable(job, 0u), SrcVertexDomain(job)}; }
    TopoSourceValues<packed_float3> SrcFaceNormals(MeshTopologyJob job) const { return {BindlessBuffer(packed_float3,B.Buffer, Pc.Source.BaseFaceNormalSlot), SrcTable(job, 2u), SrcFaceDomain(job)}; }
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
        if (!(job.Flags & (TopologyFlagSelectAll | TopologyFlagListSelects))) return Scratch()[job.SelectionOffset+v]!=0u;
        if (job.Flags & TopologyFlagSelectAll) return true;
        if (job.Flags & TopologyFlagListSelects) return (FlagVertices(job)[v] & TopoListed) != 0u;
        return Scratch()[job.SelectionOffset+v]!=0u;
    }
    bool SrcSelectedEdge(MeshTopologyJob job, uint e) const {
        if (e >= job.SrcEdgeCount) return false;
        if (!(job.Flags & (TopologyFlagSelectAll | TopologyFlagListSelects)))
            return Scratch()[job.SelectionOffset+job.SrcVertexCount+e]!=0u;
        if (job.Flags & TopologyFlagSelectAll) return true;
        if ((job.Flags & TopologyFlagListSelects) && job.Op == MeshTopologyOp::Subdivide) return EdgeParams(job)[e] != 0u;
        return Scratch()[job.SelectionOffset+job.SrcVertexCount+e]!=0u;
    }
    bool SrcSelectedFace(MeshTopologyJob job,uint f) const {
        if (f>=job.SrcFaceCount) return false;
        return (job.Flags&TopologyFlagSelectAll) || Scratch()[job.SelectionOffset+job.SrcVertexCount+job.SrcHalfedgeCount+f]!=0u;
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
    TopoOutputValues<packed_uint2> DstFaceRanges(MeshTopologyJob job) const { return {BindlessBufferMutable(packed_uint2, B.Buffer, job.DstConnectivity.FaceRangesSlot), DstTable(job, 2u), DstFaceDomain(job)}; }
    TopoOutputRun<uint> DstHalfedgeFaces(MeshTopologyJob job) const { return {BindlessBufferMutable(uint, B.Buffer, job.DstConnectivity.HalfedgeFacesSlot), DstTable(job, 1u), job.DstCornerOffset}; }
    TopoOutputRun<uint> DstOpposites(MeshTopologyJob job) const { return {BindlessBufferMutable(uint, B.Buffer, job.DstConnectivity.OppositesSlot), DstTable(job, 1u), job.DstCornerOffset}; }
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

// Named attributes travel as the words of their values, whose types use their component counts.
inline TopologyAttribute TopoAttribute(TopoContext ctx, MeshTopologyJob job, uint i) {
    return reinterpret_cast<device const TopologyAttribute *>(BindlessBuffer(uint, ctx.B.Buffer, job.Attributes.Slot) + job.Attributes.Offset)[i];
}
// The first and end records of a domain's attributes.
inline uint2 TopoAttributeRun(MeshTopologyJob job, AttributeDomain domain) {
    const uint d = uint(domain);
    return uint2(d ? job.AttributeEnds[d - 1u] : 0u, job.AttributeEnds[d]);
}
// The attribute's value at a source handle of its domain, and its default without a source value.
inline uint4 TopoReadAttribute(TopoContext ctx, TopologyAttribute attribute, uint handle) {
    if (handle == InvalidOffset || attribute.Source.ValuesSlot == InvalidSlot) return uint4(attribute.Default[0], attribute.Default[1], attribute.Default[2], attribute.Default[3]);
    return ReadNamedAttribute(ctx.B, attribute.Source, attribute.Type, handle);
}
// The operator's parameter at a source handle: the job's per-element attribute's value there, or the constant without one.
inline float TopoParamAt(TopoContext ctx, NamedAttributeRef attribute, float fallback, uint handle) {
    return attribute.ValuesSlot == InvalidSlot ? fallback : as_type<float>(ReadNamedAttribute(ctx.B, attribute, AttributeType::Float, handle).x);
}
// The first parameter at a compact source vertex or edge.
inline float TopoParam0Vertex(TopoContext ctx, MeshTopologyJob job, uint v) { return TopoParamAt(ctx, job.Param0Attribute, job.Param0, ctx.SrcVertexDomain(job).Handle(v)); }
inline float TopoParam0Edge(TopoContext ctx, MeshTopologyJob job, uint e) { return TopoParamAt(ctx, job.Param0Attribute, job.Param0, ctx.SrcEdgeDomain(job).Handle(e)); }
// Writes the attribute at an output handle of its domain.
inline void TopoWriteAttribute(TopoContext ctx, TopologyAttribute attribute, uint handle, uint4 value) {
    WriteNamedAttribute(ctx.B, attribute.Destination, attribute.Type, handle, value);
}
// Writes each attribute of the domain at an output handle from its value at a source handle.
inline void TopoCopyAttributes(TopoContext ctx, MeshTopologyJob job, AttributeDomain domain, uint handle, uint source) {
    const uint2 run = TopoAttributeRun(job, domain);
    for (uint i = run.x; i < run.y; ++i) {
        const TopologyAttribute attribute = TopoAttribute(ctx, job, i);
        TopoWriteAttribute(ctx, attribute, handle, TopoReadAttribute(ctx, attribute, source));
    }
}
// Floats interpolate bilinearly, and ints and bools take the source of the dominant weight.
inline uint4 TopoInterpolateAttribute(TopologyAttribute attribute, uint4 a, uint4 b, uint4 c, uint4 d, float s, float t) {
    if (attribute.Type == AttributeType::Int || attribute.Type == AttributeType::Bool) return t < 0.5f ? (s < 0.5f ? a : b) : (s < 0.5f ? d : c);
    return as_type<uint4>(TopoBilinear(as_type<float4>(a), as_type<float4>(b), as_type<float4>(c), as_type<float4>(d), s, t));
}

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
            return (job.Flags & TopologyFlagListSelects) ? ctx.Scratch()[job.SelectionOffset+v]!=0u : ctx.SrcSelectedVertex(job,v);
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
    const bool hidden=source<job.SrcFaceCount+job.Src2Counts[2] && (!selected || job.Op==MeshTopologyOp::KeepSelectedFaces) && ctx.Selected({ctx.Pc.Source.FaceHiddenSlot,0u},ctx.SrcAddress(job,2u,ctx.SrcFaceDomain(job).Handle(source)));
    ctx.Select({ctx.Pc.Destination.FaceHiddenSlot,0u},ctx.DstAddress(job,2u,ctx.DstFaceDomain(job).Handle(fd)),hidden);
    ctx.SelectDstFace(job, fd, selected && !hidden);
}

#endif
