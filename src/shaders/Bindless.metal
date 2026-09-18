#ifndef BINDLESS_MSL
#define BINDLESS_MSL

#include "TRSUtils.metal"
#include "gpu/BindlessBindings.h"
#include "gpu/BoneDeformVertex.h"
#include "gpu/BoundsEntry.h"
#include "gpu/CornerClassEncoding.h"
#include "gpu/DrawData.h"
#include "gpu/InstanceRecord.h"
#include "gpu/MeshRecord.h"
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

inline uint AdvancedOffset(uint offset, uint by) { return offset != InvalidOffset ? offset + by : offset; }

// The draw context: a mesh record advanced to a primitive's first triangle, with one instance's state and the selection in effect.
inline DrawData ComposeDraw(MeshRecord mesh, uint first_triangle, InstanceRecord instance, uint instance_slot, EditSelectionStorage selection) {
    const uint first_corner = first_triangle * 3u;
    return DrawData{
        .VertexSlot = mesh.VertexSlot,
        .IndexSlotOffset = {mesh.IndexSlotOffset.Slot, mesh.IndexSlotOffset.Offset + first_corner},
        .ModelSlot = mesh.ModelSlot,
        .FirstInstance = instance_slot,
        .ObjectIdSlot = mesh.ObjectIdSlot,
        .CornerClassOffset = mesh.CornerClassOffset < uint(CornerClassEncoding::UniformFaceOffset) ? mesh.CornerClassOffset + first_corner : mesh.CornerClassOffset,
        .CustomCornerMaskOffset = mesh.CustomCornerMaskOffset,
        .CustomCornerNormalOffset = mesh.CustomCornerNormalOffset,
        .CornerBase = first_corner,
        .BaseSeamNormalOffset = mesh.BaseSeamNormalOffset,
        .CornerTangentOffset = AdvancedOffset(mesh.CornerTangentOffset, first_corner),
        .CornerColorOffset = AdvancedOffset(mesh.CornerColorOffset, first_corner),
        .CornerUvOffsets = {
            AdvancedOffset(mesh.CornerUvOffsets[0], first_corner), AdvancedOffset(mesh.CornerUvOffsets[1], first_corner),
            AdvancedOffset(mesh.CornerUvOffsets[2], first_corner), AdvancedOffset(mesh.CornerUvOffsets[3], first_corner)
        },
        .FaceIdOffset = mesh.FaceIdOffset + first_triangle,
        .BaseFaceNormalOffset = mesh.BaseFaceNormalOffset,
        .FaceFirstTriangleOffset = mesh.FaceFirstTriangleOffset,
        .VertexEdgeAdjacencyOffset = mesh.VertexEdgeAdjacencyOffset,
        .VertexFanAdjacencyOffset = mesh.VertexFanAdjacencyOffset,
        .Connectivity = mesh.Connectivity,
        .EdgeHalfedges = mesh.EdgeHalfedges,
        .HalfedgeCount = mesh.HalfedgeCount,
        .FaceCount = mesh.FaceCount,
        .ConnectivityFaceStarts = mesh.ConnectivityFaceStarts,
        .VertexCountOrHeadImageSlot = mesh.VertexCountOrHeadImageSlot,
        .ElementIdOffset = instance.ElementIdOffset,
        .Selection = selection,
        .EditEdgeOffset = mesh.EditEdgeOffset,
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
        .PosedPositionOffset = instance.PosedPositionOffset,
        .PosedVertexNormalOffset = instance.PosedVertexNormalOffset,
        .PosedSeamNormalOffset = instance.PosedSeamNormalOffset,
        .PosedFaceNormalOffset = instance.PosedFaceNormalOffset,
        .PrimitiveMaterialOffset = mesh.PrimitiveMaterialOffset,
        .ElementPrimitiveOffset = mesh.ElementPrimitiveOffset,
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
    device const uint *ObjectIds(uint slot) const { return BindlessBuffer(uint, B.ObjectIdBuffer, slot); }
    device const uint *FaceFirstTriangles(uint slot) const { return BindlessBuffer(uint, B.ObjectIdBuffer, slot); }
    device const uint *Adjacency(uint slot) const { return BindlessBuffer(uint, B.Buffer, slot); }
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
    device const uint *ElementPrimitives(uint slot) const { return BindlessBuffer(uint, B.ElementPrimitiveBuffer, slot); }
    device const packed_float4 *CornerTangents(uint slot) const { return BindlessBuffer(packed_float4, B.CornerTangentBuffer, slot); }
    device const packed_float4 *CornerColors(uint slot) const { return BindlessBuffer(packed_float4, B.CornerColorBuffer, slot); }
    device const packed_float2 *CornerUvs(uint slot) const { return BindlessBuffer(packed_float2, B.CornerUvBuffer, slot); }
    device const uint *CornerClasses(uint slot) const { return BindlessBuffer(uint, B.Buffer, slot); }
    // Authored corner-normal (polar, azimuth) offsets from the derived normal, packed to the corners the mask marks present.
    device const packed_float2 *CustomCornerNormals(uint slot) const { return BindlessBuffer(packed_float2, B.Buffer, slot); }
    device const packed_uint2 *CustomCornerMasks(uint slot) const { return BindlessBuffer(packed_uint2, B.Buffer, slot); }
    device const packed_float3 *BaseSeamNormals(uint slot) const { return BindlessBuffer(packed_float3, B.Buffer, slot); }
    device const packed_float3 *BaseVertexNormals(uint slot) const { return BindlessBuffer(packed_float3, B.Buffer, slot); }
    device const packed_float3 *BaseFaceNormals(uint slot) const { return BindlessBuffer(packed_float3, B.Buffer, slot); }
    device const packed_uint2 *TileMap(uint slot) const { return BindlessBuffer(packed_uint2, B.Buffer, slot); }
    // Current-pose vertex positions in mesh-local space, and the normals derived from them.
    device const packed_float3 *PosedPositions(uint slot) const { return BindlessBuffer(packed_float3, B.Buffer, slot); }
    device const packed_float3 *PosedVertexNormals(uint slot) const { return BindlessBuffer(packed_float3, B.Buffer, slot); }
    device const packed_float3 *PosedSeamNormals(uint slot) const { return BindlessBuffer(packed_float3, B.Buffer, slot); }
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
        return ComposeDraw(MeshRecords(View.MeshRecordSlot)[instance.Mesh], 0u, instance, entry.FirstInstance, entry.Selection);
    }

    // Mesh-local vertex position: the pose pre-pass's current-pose position when the draw has one.
    float3 GetLocalPosition(DrawData draw, uint idx) const {
        return draw.PosedPositionOffset != InvalidOffset ?
            float3(PosedPositions(View.PosedPositionSlot)[draw.PosedPositionOffset + idx]) :
            float3(Vertices(draw.VertexSlot)[draw.VertexOffset + idx].Position);
    }

    // Per-vertex normal: the posed normal when the draw has one, else the base normal at the vertex-arena slot.
    float3 GetVertexNormal(DrawData draw, uint idx) const {
        return draw.PosedVertexNormalOffset != InvalidOffset ?
            float3(PosedVertexNormals(View.PosedVertexNormalSlot)[draw.PosedVertexNormalOffset + idx]) :
            float3(BaseVertexNormals(View.BaseVertexNormalSlot)[draw.VertexOffset + idx]);
    }

    // Per-face normal: the posed normal when the draw has one, else the base normal.
    float3 GetFaceNormal(DrawData draw, uint face) const {
        return draw.PosedFaceNormalOffset != InvalidOffset ?
            float3(PosedFaceNormals(View.PosedFaceNormalSlot)[draw.PosedFaceNormalOffset + face]) :
            float3(BaseFaceNormals(View.BaseFaceNormalSlot)[draw.BaseFaceNormalOffset + face]);
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
