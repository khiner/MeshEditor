#pragma once
#include "TRSUtils.metal"
#include "gpu/BindlessBindings.h"
#include "gpu/BoneDeformVertex.h"
#include "gpu/BoundsEntry.h"
#include "gpu/NormalSector.h"
#include "gpu/PoseAttributeNode.h"
#include "gpu/DrawData.h"
#include "gpu/Hash.h"
#include "gpu/InstanceRecord.h"
#include "gpu/MeshRecord.h"
#include "gpu/MeshElementBlock.h"
#include "gpu/MeshElementSet.h"
#include "gpu/MeshletInstanceFlag.h"
#include "gpu/LightRecord.h"
#include "gpu/MorphTargetVertex.h"
#include "gpu/NamedAttributeRef.h"
#include "gpu/PBRMaterial.h"
#include "gpu/SceneViewUBO.h"
#include "gpu/Vertex.h"
#include "gpu/ViewportTheme.h"
#include "gpu/WorkspaceLights.h"

constant uint STATE_SELECTED = 1u << 0;
constant uint STATE_ACTIVE = 1u << 1;
// A hidden instance keeps its slot, and every draw, pick and overlay pass skips it.
constant uint STATE_HIDDEN = 1u << 2;

inline device atomic_uint *AtomicWord(device uint *word) { return reinterpret_cast<device atomic_uint *>(word); }
inline bool MaskBit(device const uint *words, uint i) { return (words[i / 32u] >> (i % 32u)) & 1u; }
// Sets or clears bit i atomically, since neighboring bits share its word.
inline void WriteMaskBit(device uint *words, uint i, bool value) {
    device atomic_uint *word = AtomicWord(words + i / 32u);
    const uint mask = 1u << (i % 32u);
    if (value) atomic_fetch_or_explicit(word, mask, memory_order_relaxed);
    else atomic_fetch_and_explicit(word, ~mask, memory_order_relaxed);
}
// The first of `count` indices where `before` fails, for a `before` holding on a prefix of them, or `count` when it holds on all.
template<typename Before> inline uint PartitionPoint(uint count, Before before) {
    uint low = 0u, high = count;
    while (low < high) {
        const uint mid = (low + high) / 2u;
        if (before(mid)) low = mid + 1u;
        else high = mid;
    }
    return low;
}
// An open-addressed key table over a power of two of slots, a free slot holding InvalidOffset in its first word.
struct KeyTable {
    device uint *Slots;
    uint Mask;
    device uint *Slot(uint s) const { return Slots + KeyTableWords * s; }
    // Claims the first free slot along the probe from `start` for the key, writing the key's second word after the claim.
    device uint *Claim(uint2 key, uint start) const {
        for (uint s = start & Mask;; s = (s + 1u) & Mask) {
            device uint *slot = Slot(s);
            uint expected = InvalidOffset;
            if (atomic_compare_exchange_weak_explicit(AtomicWord(slot), &expected, key.x, memory_order_relaxed, memory_order_relaxed)) {
                slot[1] = key.y;
                return slot;
            }
        }
    }
    // The first slot along the probe from `start` holding the key, or null at a free slot, once every claim has written its key.
    device uint *Find(uint2 key, uint start) const {
        for (uint s = start & Mask;; s = (s + 1u) & Mask) {
            device uint *slot = Slot(s);
            if (slot[0] == InvalidOffset) return nullptr;
            if (slot[0] == key.x && slot[1] == key.y) return slot;
        }
    }
};

// A job's scratch storage, whose runs sit at word offsets and whose first word is the chain's error word.
struct JobRuns {
    device uint *Words;
    template<typename T> device T *Run(uint offset) const { return reinterpret_cast<device T *>(Words + offset); }
    // Marks the chain's error word, which fails its submit.
    void Fail() const { atomic_store_explicit(AtomicWord(Words), 1u, memory_order_relaxed); }
};
template<typename Record> struct JobContext {
    device const BindlessSet &B;
    constant JobPushConstants &Pc;
    JobRuns Storage;
    JobContext(device const BindlessSet &b, constant JobPushConstants &pc) : B(b), Pc(pc), Storage{BindlessBufferMutable(uint, b.Buffer, pc.Job.Slot)} {}
    Record Job() const { return *reinterpret_cast<device const Record *>(Storage.Words + Pc.Job.Offset); }
    template<typename T> device T *Run(uint offset) const { return Storage.Run<T>(offset); }
    void Fail() const { Storage.Fail(); }
    operator JobRuns() const { return Storage; }
};

// The arena address of a record-relative handle of a domain with `BlockSize`-slot blocks, through the record's table run.
template<uint BlockSize, typename SetT> inline uint HandleAddress(device const SetT &b, SlotOffset table, uint handle) {
    return Resolve<BlockSize>(BindlessBuffer(uint, b.Buffer, table.Slot) + table.Offset, handle);
}
// The first table entry of the arena's set at index `set`, through which the set's handles resolve, or null for an absent set.
template<typename SetT> inline device const uint *SetTable(device const SetT &b, SetArenaSlots arena, uint set) {
    return set == InvalidOffset ? nullptr : BindlessBuffer(uint, b.Buffer, arena.Table) + BindlessBuffer(MeshElementSet, b.Buffer, arena.Sets)[set].First;
}
template<typename T> struct ElementValues {
    device T *Values;
    device const uint *Table;
    uint Address(uint handle) const { return Resolve<MeshElementBlockSize>(Table, handle); }
    device T &operator[](uint handle) const { return Values[Address(handle)]; }
};
template<typename T> inline ElementValues<const T> ReadElements(device const BindlessSet &b, uint values_slot, SlotOffset table) {
    return {BindlessBuffer(T, b.Buffer, values_slot), BindlessBuffer(uint, b.Buffer, table.Slot) + table.Offset};
}
template<typename T> inline ElementValues<T> WriteElements(device const BindlessSet &b, uint values_slot, SlotOffset table) {
    return {BindlessBufferMutable(T, b.Buffer, values_slot), BindlessBuffer(uint, b.Buffer, table.Slot) + table.Offset};
}


// The value index of a morph target's delta at a vertex handle, where each vertex block's targets are consecutive blocks of the record's range.
inline uint MorphTargetIndex(SlotOffset morph, uint handle, uint target, uint targets) {
    return morph.Offset + (handle / MeshElementBlockSize * targets + target) * MeshElementBlockSize + handle % MeshElementBlockSize;
}
inline uint4 ReadNamedAttribute(device const BindlessSet &b, NamedAttributeRef attribute, AttributeType type, uint handle) {
    const uint index = attribute.Offset + handle;
    device const uint *words = BindlessBuffer(uint, b.Buffer, attribute.ValuesSlot);
    if (type == AttributeType::Bool) return uint4(MaskBit(words, index) ? 1u : 0u, 0u, 0u, 0u);
    const uint n = AttributeComponents(type);
    uint4 value = uint4(0u);
    for (uint c = 0u; c < n; ++c) value[c] = words[index * n + c];
    return value;
}
inline void WriteNamedAttribute(device const BindlessSet &b, NamedAttributeRef attribute, AttributeType type, uint handle, uint4 value) {
    const uint index = attribute.Offset + handle;
    device uint *words = BindlessBufferMutable(uint, b.Buffer, attribute.ValuesSlot);
    if (type == AttributeType::Bool) {
        WriteMaskBit(words, index, value.x != 0u);
        return;
    }
    const uint n = AttributeComponents(type);
    for (uint c = 0u; c < n; ++c) words[index * n + c] = value[c];
}

// The index of the posed value at arena address `record` in the pose's two-level radix tree at `root`, or InvalidOffset when the pose has none.
inline uint PoseAttributeIndex(device const BindlessSet &b, uint nodes_slot, uint root, uint record) {
    if (root == InvalidOffset) return InvalidOffset;
    device const PoseAttributeNode *nodes = BindlessBuffer(PoseAttributeNode,b.Buffer,nodes_slot);
    const uint child = nodes[root].Children[record >> (8u + PoseAttributeRadixBits)];
    if (!child) return InvalidOffset;
    const uint value = nodes[child-1u].Children[(record >> 8u)&PoseAttributeRadixMask];
    return value ? (value-1u)*256u+(record&255u) : InvalidOffset;
}

// The sector root handle of a corner handle, or InvalidOffset for a corner without one or a record without sector roots.
inline uint CornerSectorRoot(device const BindlessSet &b, SlotOffset sectors, uint h) {
    return sectors.Slot != InvalidSlot ? BindlessBuffer(uint, b.Buffer, sectors.Slot)[sectors.Offset + h] : InvalidOffset;
}

inline uint TriangleCornerHandle(device const BindlessSet &b, uint slot, ConnectivityRef connectivity, uint first, uint corner) {
    return BindlessBuffer(packed_uint3, b.Buffer, slot)[HandleAddress<MeshElementBlockSize>(b, connectivity.Tables.Triangles, first + corner / 3u)][corner % 3u];
}
inline uint TriangleFaceHandle(device const BindlessSet &b, ConnectivityRef connectivity, uint slot, uint triangle) {
    const uint h = BindlessBuffer(packed_uint3, b.Buffer, slot)[HandleAddress<MeshElementBlockSize>(b, connectivity.Tables.Triangles, triangle)].x;
    return BindlessBuffer(uint, b.Buffer, connectivity.HalfedgeFacesSlot)[HandleAddress<MeshElementBlockSize>(b, connectivity.Tables.Halfedges, h)];
}

// The pose namespace an instance draws with: its own under per-instance deformation, else its mesh's shared one.
inline uint PoseNamespace(uint instance_namespace, uint mesh_namespace) {
    return instance_namespace != InvalidOffset ? instance_namespace : mesh_namespace;
}

// The flags an instance draws with: its mesh's, without the primary-only flags on other instances, plus Silhouette from its selection state.
inline uint InstanceFlags(uint mesh_flags, uint primary_edit_instance, uint instance_slot, uint instance_state) {
    const uint primary_only = uint(MeshletInstanceFlag::ElementSelection) | uint(MeshletInstanceFlag::EditOverlay);
    const bool primary = primary_edit_instance == instance_slot;
    const bool silhouette = (instance_state & STATE_SELECTED) != 0u && !primary &&
        (mesh_flags & uint(MeshletInstanceFlag::SilhouetteEligible)) != 0u;
    return (primary ? mesh_flags : mesh_flags & ~primary_only) | (silhouette ? uint(MeshletInstanceFlag::Silhouette) : 0u);
}

// Absent selection uses InvalidSlot because zero is a valid buffer slot.
constant EditSelectionStorage NoEditSelection{{InvalidSlot, 0u}, {InvalidSlot, 0u}, {InvalidSlot, 0u}, {InvalidSlot, 0u}, InvalidSlot, InvalidSlot, InvalidSlot};

// The mesh's primary edit instance draws its element selection, and every instance does when it has none.
inline DrawData ComposeDraw(MeshRecord mesh, InstanceRecord instance, uint instance_slot, uint topology = 0u) {
    const MeshDisplay display = mesh.Display;
    const bool selection = display.PrimaryEditInstanceIndex == InvalidOffset || display.PrimaryEditInstanceIndex == instance_slot;
    const bool faces = topology == 0u && mesh.TriangleSlot != InvalidSlot;
    const ElementLayers layers = mesh.Layers;
    const SlotOffset absent{InvalidSlot, 0u};
    return DrawData{
        .VertexSlot = mesh.VertexSlot,
        .IndexSlotOffset = mesh.IndexSlotOffset,
        .ModelSlot = mesh.ModelSlot,
        .FirstInstance = instance_slot,
        .TriangleSlot = faces ? mesh.TriangleSlot : InvalidSlot,
        .CornerClassMode = mesh.CornerClassMode,
        .CustomNormals = faces ? layers.CustomNormals : absent,
        .CornerTangent = faces ? layers.CornerTangent : absent,
        .CornerColor = faces ? layers.CornerColor : layers.VertexColor,
        .CornerUvs = faces ? layers.CornerUvs : GpuArray<SlotOffset, 4>{absent, absent, absent, absent},
        .Connectivity = mesh.Connectivity,
        .FaceCount = mesh.FaceCount,
        .Selection = selection ? display.Selection : NoEditSelection,
        .BoneDeformOffset = display.BoneDeformOffset,
        .ArmatureDeformOffset = instance.ArmatureDeformOffset,
        .MorphDeformOffset = display.MorphDeformOffset,
        .MorphWeightsOffset = instance.MorphWeightsOffset,
        .MorphTargetCount = display.MorphTargetCount,
        .MorphShadingAuthored = mesh.MorphShadingAuthored != 0u && display.MorphDeformOffset != InvalidOffset ? 1u : 0u,
        .PositionNamespace = PoseNamespace(instance.PositionNamespace, display.PositionNamespace),
        .MorphNormalNamespace = PoseNamespace(instance.MorphNormalNamespace, display.MorphNormalNamespace),
        .VertexNormalNamespace = PoseNamespace(instance.VertexNormalNamespace, display.VertexNormalNamespace),
        .SectorNamespace = PoseNamespace(instance.SectorNamespace, display.SectorNamespace),
        .FaceNormalNamespace = PoseNamespace(instance.FaceNormalNamespace, display.FaceNormalNamespace),
        .PrimitiveMaterialOffset = mesh.PrimitiveMaterialOffset,
        .ElementPrimitives = faces ? layers.FacePrimitives : layers.VertexPrimitives,
        .Skin = layers.Skin,
        .Morph = layers.Morph,
        .CornerSectors = layers.CornerSectors,
        .NormalSectors = layers.NormalSectors,
    };
}

// Provides resources outside an entry point's explicit parameters because MSL has no global resource bindings.
// The table uses device address space because it exceeds constant-space limits and contains device addresses.
template<typename SetT>
struct SceneT {
    device const SetT &B;
    constant SceneViewUBO &View;
    constant ViewportThemeUbo &Theme;
    constant WorkspaceLights &Workspace;

    // Projection matrices use Metal's positive-Y-up clip space.
    float4x4 ViewProj() const { return View.ViewProj.Unpack(); }

    device const Vertex *Vertices(uint slot) const { return BindlessBuffer(Vertex, B.VertexBuffer, slot); }
    device const Transform *Models(uint slot) const { return BindlessBuffer(Transform, B.ModelBuffer, slot); }
    device const uint *Indices(uint slot) const { return BindlessBuffer(uint, B.IndexBuffer, slot); }
    device const uchar *Bytes(uint slot) const { return BindlessBuffer(uchar, B.Buffer, slot); }
    device const uint *FaceTriangles(uint slot) const { return BindlessBuffer(uint, B.ObjectIdBuffer, slot); }
    device const BoundsEntry *BoundsEntries(uint slot) const { return BindlessBuffer(BoundsEntry, B.BoundsEntryBuffer, slot); }
    device const MeshRecord *MeshRecords(uint slot) const { return BindlessBuffer(MeshRecord, B.Buffer, slot); }
    device const uchar *InstanceStates(uint slot) const { return BindlessBuffer(uchar, B.InstanceStateBuffer, slot); }
    device const BoneDeformVertex *BoneDeforms(uint slot) const { return BindlessBuffer(BoneDeformVertex, B.BoneDeformBuffer, slot); }
    device const mat4 *ArmatureDeforms(uint slot) const { return BindlessBuffer(mat4, B.ArmatureDeformBuffer, slot); }
    device const MorphTargetVertex *MorphTargets(uint slot) const { return BindlessBuffer(MorphTargetVertex, B.MorphTargetBuffer, slot); }
    device const float *MorphWeights(uint slot) const { return BindlessBuffer(float, B.MorphWeightBuffer, slot); }
    device const LightRecord *Lights(uint slot) const { return BindlessBuffer(LightRecord, B.LightBuffer, slot); }
    device const PBRMaterial *Materials(uint slot) const { return BindlessBuffer(PBRMaterial, B.MaterialBuffer, slot); }
    device const uint *PrimitiveMaterials(uint slot) const { return BindlessBuffer(uint, B.PrimitiveMaterialBuffer, slot); }
    uint VertexAddress(DrawData draw, uint v) const { return HandleAddress<MeshElementBlockSize>(B, draw.Connectivity.Tables.Vertices, v); }
    uint CornerAddress(DrawData draw, uint h) const { return HandleAddress<MeshElementBlockSize>(B, draw.Connectivity.Tables.Halfedges, h); }
    uint EdgeAddress(DrawData draw, uint e) const { return HandleAddress<MeshElementBlockSize>(B, draw.Connectivity.Tables.Edges, e); }
    uint FaceAddress(DrawData draw, uint f) const { return HandleAddress<MeshElementBlockSize>(B, draw.Connectivity.Tables.Faces, f); }
    uint ElementPrimitive(SlotOffset primitives, uint handle) const { return BindlessBuffer(uint, B.ElementPrimitiveBuffer, primitives.Slot)[primitives.Offset + handle]; }
    uint CornerVertex(DrawData draw, uint corner) const {
        return Indices(draw.IndexSlotOffset.Slot)[CornerAddress(draw, corner)];
    }
    uint ExtrasVertex(DrawData draw, uint index) const {
        return Indices(draw.IndexSlotOffset.Slot)[draw.IndexSlotOffset.Offset + index];
    }
    uint CornerFace(DrawData draw, uint corner) const {
        return BindlessBuffer(uint, B.Buffer, draw.Connectivity.HalfedgeFacesSlot)[CornerAddress(draw, corner)];
    }
    uint TriangleFace(DrawData draw, uint triangle) const {
        return TriangleFaceHandle(B, draw.Connectivity, draw.TriangleSlot, triangle);
    }
    // Corner layers at a corner's handle, or a point or line draw's color at a vertex handle.
    float4 CornerTangent(SlotOffset at, uint h) const { return float4(BindlessBuffer(packed_float4, B.CornerTangentBuffer, at.Slot)[at.Offset + h]); }
    float4 CornerColor(SlotOffset at, uint h) const { return float4(BindlessBuffer(packed_float4, B.CornerColorBuffer, at.Slot)[at.Offset + h]); }
    float2 CornerUv(SlotOffset at, uint h) const { return float2(BindlessBuffer(packed_float2, B.CornerUvBuffer, at.Slot)[at.Offset + h]); }

    // Weight-summed authored morph normal deltas, indexed like the posed positions.
    device const packed_float3 *PosedMorphNormalDeltas(uint slot) const { return BindlessBuffer(packed_float3, B.Buffer, slot); }

    float4 SampleTex(uint slot, float2 uv) const { return B.Sampler[slot].Texture.sample(B.Sampler[slot].Sampler, uv); }
    float4 SampleTexGrad(uint slot, float2 uv, float2 dx, float2 dy) const {
        return B.Sampler[slot].Texture.sample(B.Sampler[slot].Sampler, uv, gradient2d(dx, dy));
    }
    float4 SampleTexLod(uint slot, float2 uv, float lod) const { return B.Sampler[slot].Texture.sample(B.Sampler[slot].Sampler, uv, level(lod)); }
    float4 FetchTex(uint slot, int2 px, uint lod) const { return B.Sampler[slot].Texture.read(uint2(px), lod); }
    uint2 TexSize(uint slot, uint lod) const { return uint2(B.Sampler[slot].Texture.get_width(lod), B.Sampler[slot].Texture.get_height(lod)); }
    float4 SampleCube(uint slot, float3 dir) const { return B.CubeSampler[slot].Texture.sample(B.CubeSampler[slot].Sampler, dir); }
    float4 SampleCubeLod(uint slot, float3 dir, float lod) const { return B.CubeSampler[slot].Texture.sample(B.CubeSampler[slot].Sampler, dir, level(lod)); }

    device const InstanceRecord *InstanceRecords(uint slot) const { return BindlessBuffer(InstanceRecord, B.Buffer, slot); }

    // The draw context of a bounds entry: its first instance's mesh and deform state with the mesh's edit selection.
    DrawData BoundsDraw(BoundsEntry entry) const {
        const InstanceRecord instance = InstanceRecords(View.InstanceRecordSlot)[entry.FirstInstance];
        DrawData draw = ComposeDraw(MeshRecords(View.MeshRecordSlot)[instance.Mesh], instance, entry.FirstInstance);
        draw.Selection = entry.Selection;
        return draw;
    }

    // Mesh-local position at a vertex's arena address: the pose pre-pass's current-pose position when the draw has one.
    float3 LocalPositionAt(DrawData draw, uint address) const {
        const uint posed = PoseAttributeIndex(B, View.PosedPositionNodesSlot, draw.PositionNamespace, address);
        return posed != InvalidOffset ? float3(BindlessBuffer(packed_float3, B.Buffer, View.PosedPositionSlot)[posed]) :
            float3(Vertices(draw.VertexSlot)[address].Position);
    }
    float3 GetLocalPosition(DrawData draw, uint v) const { return LocalPositionAt(draw, VertexAddress(draw, v)); }

    float3 GetVertexNormal(DrawData draw, uint v) const {
        const uint address = VertexAddress(draw, v);
        const uint posed = PoseAttributeIndex(B, View.PosedVertexNormalNodesSlot, draw.VertexNormalNamespace, address);
        return posed != InvalidOffset ? float3(BindlessBuffer(packed_float3, B.Buffer, View.PosedVertexNormalSlot)[posed]) :
            float3(BindlessBuffer(packed_float3, B.Buffer, View.BaseVertexNormalSlot)[address]);
    }

    float3 GetFaceNormal(DrawData draw, uint face) const {
        const uint address = FaceAddress(draw, face);
        const uint posed = PoseAttributeIndex(B, View.PosedFaceNormalNodesSlot, draw.FaceNormalNamespace, address);
        return posed != InvalidOffset ? float3(BindlessBuffer(packed_float3, B.Buffer, View.PosedFaceNormalSlot)[posed]) :
            float3(BindlessBuffer(packed_float3, B.Buffer, View.BaseFaceNormalSlot)[address]);
    }

    // A pick id is an element's arena address plus one, unique across records, and zero means no hit.
    uint FacePickId(DrawData draw, uint face) const { return face == InvalidOffset ? 0u : FaceAddress(draw, face) + 1u; }
    uint ElementPickId(DrawData draw, bool lines, uint element) const { return (lines ? EdgeAddress(draw, element) : VertexAddress(draw, element)) + 1u; }

    uint InstanceState(DrawData draw) const {
        return uint(InstanceStates(View.InstanceStateSlot)[draw.FirstInstance]);
    }

    float4 ObjectSelectionColor(uint instance_state, float4 unselected) const {
        if ((instance_state & STATE_SELECTED) == 0u) return unselected;
        const bool is_active = (instance_state & STATE_ACTIVE) != 0u;
        return float4(is_active ? float3(Theme.Colors.ObjectActive) : float3(Theme.Colors.ObjectSelected), 1.0f);
    }
};

using Scene = SceneT<BindlessSet>;
using SceneImageWrite = SceneT<BindlessSetImageWrite>;

inline float3 NormalizeOrZero(float3 n) {
    const float len = length(n);
    return len > 0.0f ? n / len : float3(0.0f);
}
