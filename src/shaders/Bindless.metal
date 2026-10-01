#ifndef BINDLESS_MSL
#define BINDLESS_MSL

#include "TRSUtils.metal"
#include "gpu/BindlessBindings.h"
#include "gpu/BoneDeformVertex.h"
#include "gpu/BoundsEntry.h"
#include "gpu/CornerClassMode.h"
#include "gpu/NormalSector.h"
#include "gpu/PoseAttributeNode.h"
#include "gpu/DrawData.h"
#include "gpu/InstanceRecord.h"
#include "gpu/MeshRecord.h"
#include "gpu/MeshElementBlock.h"
#include "gpu/LightRecord.h"
#include "gpu/MorphTargetVertex.h"
#include "gpu/PBRMaterial.h"
#include "gpu/SceneViewUBO.h"
#include "gpu/Transform.h"
#include "gpu/Vertex.h"
#include "gpu/ViewportTheme.h"
#include "gpu/WorkspaceLights.h"

constant uint STATE_SELECTED = 1u << 0;
constant uint STATE_ACTIVE = 1u << 1;

// Select a live canonical handle by its ordinal within one element block.
inline uint SelectLiveElement(device const uint *live, uint block, uint rank) {
    for (uint word = 0u; word < MeshElementBlockWords; ++word) {
        uint bits = live[word];
        const uint count = popcount(bits);
        if (rank >= count) { rank -= count; continue; }
        while (rank) { bits &= bits - 1u; --rank; }
        return block * MeshElementBlockSize + word * 32u + ctz(bits);
    }
    return InvalidOffset;
}

// The value index of entry `entry` of `handle`, whose element block owns consecutive payload blocks named by `blocks`.
inline uint ElementAttributeIndex(device const uint *blocks, uint handle, uint entry = 0u) {
    return (blocks[handle / MeshElementBlockSize] - 1u + entry) * MeshElementBlockSize + handle % MeshElementBlockSize;
}
inline uint ElementAttributeIndex(device const BindlessSet &b, ElementAttributeRef attribute, uint handle, uint entry = 0u) {
    return ElementAttributeIndex(BindlessBuffer(uint, b.Buffer, attribute.BlocksSlot), handle, entry);
}

// Each pose maps canonical record keys into shared typed value blocks.
inline uint PoseAttributeIndex(device const BindlessSet &b, uint nodes_slot, uint root, uint record) {
    if (root == InvalidOffset) return InvalidOffset;
    device const PoseAttributeNode *nodes = BindlessBuffer(PoseAttributeNode,b.Buffer,nodes_slot);
    const uint child = nodes[root].Children[record >> (8u + PoseAttributeRadixBits)];
    if (!child) return InvalidOffset;
    const uint value = nodes[child-1u].Children[(record >> 8u)&PoseAttributeRadixMask];
    return value ? (value-1u)*256u+(record&255u) : InvalidOffset;
}

inline uint CornerSectorRoot(device const BindlessSet &b, ElementAttributeRef attribute, uint handle) {
    const uint block = BindlessBuffer(uint, b.Buffer, attribute.BlocksSlot)[handle / MeshElementBlockSize];
    return block ? BindlessBuffer(uint, b.Buffer, attribute.ValuesSlot)[(block - 1u) * MeshElementBlockSize + handle % MeshElementBlockSize] : InvalidOffset;
}

inline uint TriangleCornerHandle(device const BindlessSet &b, uint slot, uint corner, uint triangle_base) {
    return BindlessBuffer(packed_uint3, b.Buffer, slot)[triangle_base + corner / 3u][corner % 3u];
}
inline uint TriangleFaceHandle(device const BindlessSet &b, ConnectivityRef connectivity, uint slot, uint triangle) {
    const uint h = BindlessBuffer(packed_uint3, b.Buffer, slot)[triangle].x;
    return BindlessBuffer(uint, b.Buffer, connectivity.HalfedgeFaces.Slot)[h];
}

// Compose mesh and instance state without rebasing canonical references.
inline DrawData ComposeDraw(MeshRecord mesh, InstanceRecord instance, uint instance_slot, EditSelectionStorage selection) {
    return DrawData{
        .VertexSlot = mesh.VertexSlot,
        .IndexSlotOffset = mesh.IndexSlotOffset,
        .ModelSlot = mesh.ModelSlot,
        .FirstInstance = instance_slot,
        .TriangleSlot = mesh.TriangleSlot,
        .CornerClassMode = mesh.CornerClassMode,
        .CustomNormals = mesh.CustomNormals,
        .CornerTangent = mesh.CornerTangent,
        .CornerColor = mesh.CornerColor,
        .CornerUvs = mesh.CornerUvs,
        .Connectivity = mesh.Connectivity,
        .HalfedgeCount = mesh.HalfedgeCount,
        .FaceCount = mesh.FaceCount,
        .VertexCountOrHeadImageSlot = mesh.VertexCountOrHeadImageSlot,
        .ElementIdOffset = instance.ElementIdOffset,
        .Selection = selection,
        .InstanceStateSlot = mesh.InstanceStateSlot,
        .HasPendingVertexTransform = instance.HasPendingVertexTransform,
        .PrimaryEditInstanceIndex = instance.PrimaryEditInstanceIndex,
        .VertexOffset = mesh.VertexOffset,
        .BoneDeformOffset = instance.BoneDeformOffset,
        .ArmatureDeformOffset = instance.ArmatureDeformOffset,
        .MorphDeformOffset = instance.MorphDeformOffset,
        .MorphWeightsOffset = instance.MorphWeightsOffset,
        .MorphTargetCount = instance.MorphTargetCount,
        .MorphShadingAuthored = mesh.MorphShadingAuthored != 0u && instance.MorphDeformOffset != InvalidOffset ? 1u : 0u,
        .PositionNamespace = instance.PositionNamespace,
        .MorphNormalNamespace = instance.MorphNormalNamespace,
        .VertexNormalNamespace = instance.VertexNormalNamespace,
        .SectorNamespace = instance.SectorNamespace,
        .FaceNormalNamespace = instance.FaceNormalNamespace,
        .PrimitiveMaterialOffset = mesh.PrimitiveMaterialOffset,
        .ElementPrimitives = mesh.ElementPrimitives,
    };
}

// Provides resources outside an entry point's explicit parameters because MSL has no global resource bindings.
// The table uses device address space because it exceeds constant-space limits and contains device addresses.
// Image-writing shaders instantiate the same layout with a writable image type.
template<typename SetT>
struct SceneT {
    device const SetT &B;
    constant SceneViewUBO &View;
    constant ViewportTheme &Theme;
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
    uint ElementPrimitive(ElementAttributeRef attribute, uint handle) const {
        return BindlessBuffer(uint, B.ElementPrimitiveBuffer, attribute.ValuesSlot)[ElementAttributeIndex(B, attribute, handle)];
    }
    uint CornerVertexOrdinal(DrawData draw, uint handle) const {
        return Indices(draw.IndexSlotOffset.Slot)[handle] - draw.VertexOffset;
    }
    uint CornerFace(DrawData draw, uint handle) const {
        return BindlessBuffer(uint, B.Buffer, draw.Connectivity.HalfedgeFaces.Slot)[handle];
    }
    uint TriangleFace(DrawData draw, uint triangle) const {
        return TriangleFaceHandle(B, draw.Connectivity, draw.TriangleSlot, triangle);
    }
    float4 CornerTangent(ElementAttributeRef at, uint h) const { return float4(BindlessBuffer(packed_float4, B.CornerTangentBuffer, at.ValuesSlot)[ElementAttributeIndex(B, at, h)]); }
    float4 CornerColor(ElementAttributeRef at, uint h) const { return float4(BindlessBuffer(packed_float4, B.CornerColorBuffer, at.ValuesSlot)[ElementAttributeIndex(B, at, h)]); }
    float2 CornerUv(ElementAttributeRef at, uint h) const { return float2(BindlessBuffer(packed_float2, B.CornerUvBuffer, at.ValuesSlot)[ElementAttributeIndex(B, at, h)]); }

    device const packed_float3 *BaseVertexNormals(uint slot) const { return BindlessBuffer(packed_float3, B.Buffer, slot); }
    device const packed_float3 *BaseFaceNormals(uint slot) const { return BindlessBuffer(packed_float3, B.Buffer, slot); }
    // Current-pose vertex positions in mesh-local space, and the normals derived from them.
    device const packed_float3 *PosedPositions(uint slot) const { return BindlessBuffer(packed_float3, B.Buffer, slot); }
    device const packed_float3 *PosedVertexNormals(uint slot) const { return BindlessBuffer(packed_float3, B.Buffer, slot); }
    device const packed_float3 *PosedFaceNormals(uint slot) const { return BindlessBuffer(packed_float3, B.Buffer, slot); }
    // Weight-summed authored morph normal deltas, indexed like the posed positions.
    device const packed_float3 *PosedMorphNormalDeltas(uint slot) const { return BindlessBuffer(packed_float3, B.Buffer, slot); }

    // A sampler slot combines its texture and sampler IDs.
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
        return ComposeDraw(MeshRecords(View.MeshRecordSlot)[instance.Mesh], instance, entry.FirstInstance, entry.Selection);
    }

    // Mesh-local vertex position: the pose pre-pass's current-pose position when the draw has one.
    float3 GetLocalPosition(DrawData draw, uint idx) const {
        const uint vertex_id = draw.VertexOffset + idx;
        const uint posed = PoseAttributeIndex(B, View.PosedPositionNodesSlot, draw.PositionNamespace, vertex_id);
        return posed != InvalidOffset ? float3(PosedPositions(View.PosedPositionSlot)[posed]) :
            float3(Vertices(draw.VertexSlot)[vertex_id].Position);
    }

    // Per-vertex normal: the posed normal when the draw has one, else the base normal at the vertex-arena slot.
    float3 GetVertexNormal(DrawData draw, uint idx) const {
        const uint handle = draw.VertexOffset + idx;
        const uint posed = PoseAttributeIndex(B, View.PosedVertexNormalNodesSlot, draw.VertexNormalNamespace, handle);
        return posed != InvalidOffset ? float3(PosedVertexNormals(View.PosedVertexNormalSlot)[posed]) :
            float3(BaseVertexNormals(View.BaseVertexNormalSlot)[handle]);
    }

    // Canonical face handle, with a sparse pose override when present.
    float3 GetFaceNormal(DrawData draw, uint face) const {
        const uint posed = PoseAttributeIndex(B, View.PosedFaceNormalNodesSlot, draw.FaceNormalNamespace, face);
        return posed != InvalidOffset ? float3(PosedFaceNormals(View.PosedFaceNormalSlot)[posed]) :
            float3(BaseFaceNormals(View.BaseFaceNormalSlot)[face]);
    }

    // Picking still addresses packed selection masks.
    // Zero is its no-hit value.
    uint FacePickId(DrawData draw, uint face) const {
        return draw.ElementIdOffset + (face == InvalidOffset ? 0u : face - draw.Connectivity.FaceRanges.Offset + 1u);
    }

    uint InstanceState(DrawData draw) const {
        return draw.InstanceStateSlot != InvalidSlot ?
            uint(InstanceStates(draw.InstanceStateSlot)[draw.FirstInstance]) :
            0u;
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

#endif
