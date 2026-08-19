#ifndef BINDLESS_MSL
#define BINDLESS_MSL

#include "BindlessBindings.metal"
#include "SceneViewUBO.metal"
#include "ViewportTheme.metal"
#include "WorkspaceLights.metal"
#include "DrawData.metal"
#include "Vertex.metal"
#include "BoneDeformVertex.metal"
#include "MorphTargetVertex.metal"
#include "PBRMaterial.metal"
#include "PunctualLight.metal"
#include "Transform.metal"
#include "TRSUtils.metal"

constant uint INVALID_SLOT = 0xffffffffu;
constant uint INVALID_OFFSET = 0xffffffffu;
constant uint STATE_SELECTED = 1u << 0;
constant uint STATE_ACTIVE = 1u << 1;
constant uint STATE_EXCITED = 1u << 2;

// What a shader reaches beyond its own parameters. MSL has no global resource bindings, so a shader
// builds one of these at its entry point and passes it down.
// The table is reached in the device address space: it outgrows the constant-space limit, and its
// entries are device addresses, which cannot be cast into constant space. The set type is a
// parameter so a shader writing storage images picks that view over the same layout.
template<typename SetT>
struct SceneT {
    device const SetT &B;
    constant SceneViewUBO &View;
    constant ViewportTheme &Theme;
    constant WorkspaceLights &Workspace;

    // The projections put +Y up in clip space, which is what Metal rasterizes against directly.
    float4x4 ViewProj() const { return View.ViewProj.Unpack(); }
    float4x4 PrevViewProj() const { return View.PrevViewProj.Unpack(); }
    float4x4 NextViewProj() const { return View.NextViewProj.Unpack(); }

    device const Vertex *Vertices(uint slot) const { return BindlessBuffer(Vertex, B.VertexBuffer, slot); }
    device const Transform *Models(uint slot) const { return BindlessBuffer(Transform, B.ModelBuffer, slot); }
    device const uint *Indices(uint slot) const { return BindlessBuffer(uint, B.IndexBuffer, slot); }
    device const uchar *ElementStates(uint slot) const { return BindlessBuffer(uchar, B.Buffer, slot); }
    device const uint *ObjectIds(uint slot) const { return BindlessBuffer(uint, B.ObjectIdBuffer, slot); }
    device const DrawData *Draws(uint slot) const { return BindlessBuffer(DrawData, B.DrawDataBuffer, slot); }
    device const uint *VisibleIndices(uint slot) const { return BindlessBuffer(uint, B.Buffer, slot); }
    device const uchar *InstanceStates(uint slot) const { return BindlessBuffer(uchar, B.InstanceStateBuffer, slot); }
    device const BoneDeformVertex *BoneDeforms(uint slot) const { return BindlessBuffer(BoneDeformVertex, B.BoneDeformBuffer, slot); }
    device const packed_float4x4 *ArmatureDeforms(uint slot) const { return BindlessBuffer(packed_float4x4, B.ArmatureDeformBuffer, slot); }
    device const MorphTargetVertex *MorphTargets(uint slot) const { return BindlessBuffer(MorphTargetVertex, B.MorphTargetBuffer, slot); }
    device const float *MorphWeights(uint slot) const { return BindlessBuffer(float, B.MorphWeightBuffer, slot); }
    device const uchar *VertexClasses(uint slot) const { return BindlessBuffer(uchar, B.VertexClassBuffer, slot); }
    device const PunctualLight *Lights(uint slot) const { return BindlessBuffer(PunctualLight, B.LightBuffer, slot); }
    device const PBRMaterial *Materials(uint slot) const { return BindlessBuffer(PBRMaterial, B.MaterialBuffer, slot); }
    device const uint *PrimitiveMaterials(uint slot) const { return BindlessBuffer(uint, B.PrimitiveMaterialBuffer, slot); }
    device const uint *ElementPrimitives(uint slot) const { return BindlessBuffer(uint, B.ElementPrimitiveBuffer, slot); }
    device const packed_float4 *CornerTangents(uint slot) const { return BindlessBuffer(packed_float4, B.CornerTangentBuffer, slot); }
    device const packed_float4 *CornerColors(uint slot) const { return BindlessBuffer(packed_float4, B.CornerColorBuffer, slot); }
    device const packed_float2 *CornerUvs(uint slot) const { return BindlessBuffer(packed_float2, B.CornerUvBuffer, slot); }
    device const uint *CornerClasses(uint slot) const { return BindlessBuffer(uint, B.Buffer, slot); }
    // Authored corner-normal (polar, azimuth) offsets from the derived normal, packed to the corners the mask marks present.
    device const packed_float2 *CustomCornerNormals(uint slot) const { return BindlessBuffer(packed_float2, B.Buffer, slot); }
    // One (bitset word, exclusive rank) pair per 32 mesh corners.
    device const packed_uint2 *CustomCornerMasks(uint slot) const { return BindlessBuffer(packed_uint2, B.Buffer, slot); }
    // Composed sector normal per seam corner, for static face draws.
    device const packed_float3 *BaseSeamNormals(uint slot) const { return BindlessBuffer(packed_float3, B.Buffer, slot); }
    // Derived smooth normals for triangle meshes, authored normals for face-less meshes.
    device const packed_float3 *BaseVertexNormals(uint slot) const { return BindlessBuffer(packed_float3, B.Buffer, slot); }
    device const packed_float3 *BaseFaceNormals(uint slot) const { return BindlessBuffer(packed_float3, B.Buffer, slot); }
    // One element per threadgroup, (entry index, tile index) over 256-element tiles.
    device const packed_uint2 *TileMap(uint slot) const { return BindlessBuffer(packed_uint2, B.Buffer, slot); }
    // Current-pose vertex positions in mesh-local space, and the normals derived from them.
    device const packed_float3 *PosedPositions(uint slot) const { return BindlessBuffer(packed_float3, B.Buffer, slot); }
    device const packed_float3 *PosedVertexNormals(uint slot) const { return BindlessBuffer(packed_float3, B.Buffer, slot); }
    device const packed_float3 *PosedSeamNormals(uint slot) const { return BindlessBuffer(packed_float3, B.Buffer, slot); }
    device const packed_float3 *PosedFaceNormals(uint slot) const { return BindlessBuffer(packed_float3, B.Buffer, slot); }
    // Weight-summed authored morph normal deltas, indexed like the posed positions.
    device const packed_float3 *PosedMorphNormalDeltas(uint slot) const { return BindlessBuffer(packed_float3, B.Buffer, slot); }

    // A sampler slot holds the texture and the sampler, so each read names the slot once.
    float4 SampleTex(uint slot, float2 uv) const { return B.Sampler[slot].Texture.sample(B.Sampler[slot].Sampler, uv); }
    float4 SampleTexLod(uint slot, float2 uv, float lod) const { return B.Sampler[slot].Texture.sample(B.Sampler[slot].Sampler, uv, level(lod)); }
    float4 FetchTex(uint slot, int2 px, uint lod) const { return B.Sampler[slot].Texture.read(uint2(px), lod); }
    uint2 TexSize(uint slot, uint lod) const { return uint2(B.Sampler[slot].Texture.get_width(lod), B.Sampler[slot].Texture.get_height(lod)); }
    float4 SampleCube(uint slot, float3 dir) const { return B.CubeSampler[slot].Texture.sample(B.CubeSampler[slot].Sampler, dir); }
    float4 SampleCubeLod(uint slot, float3 dir, float lod) const { return B.CubeSampler[slot].Texture.sample(B.CubeSampler[slot].Sampler, dir, level(lod)); }

    // Mesh-local vertex position: the pose pre-pass's current-pose position when the draw has one.
    float3 GetLocalPosition(DrawData draw, uint idx) const {
        return draw.PosedPositionOffset != INVALID_OFFSET ?
            float3(PosedPositions(View.PosedPositionSlot)[draw.PosedPositionOffset + idx]) :
            float3(Vertices(draw.VertexSlot)[draw.VertexOffset + idx].Position);
    }

    // Per-vertex normal: the posed normal when the draw has one, else the base normal at the vertex-arena slot.
    float3 GetVertexNormal(DrawData draw, uint idx) const {
        return draw.PosedVertexNormalOffset != INVALID_OFFSET ?
            float3(PosedVertexNormals(View.PosedVertexNormalSlot)[draw.PosedVertexNormalOffset + idx]) :
            float3(BaseVertexNormals(View.BaseVertexNormalSlot)[draw.VertexOffset + idx]);
    }

    // Selection state of the instance this draw renders.
    uint InstanceState(DrawData draw) const {
        return draw.InstanceStateSlot != INVALID_SLOT ?
            uint(InstanceStates(draw.InstanceStateSlot)[draw.FirstInstance]) :
            0u;
    }

    // An object's selection color, falling back to `unselected` while it is not selected.
    float4 ObjectSelectionColor(uint instance_state, float4 unselected) const {
        if ((instance_state & STATE_SELECTED) == 0u) return unselected;
        const bool is_active = (instance_state & STATE_ACTIVE) != 0u;
        return float4(is_active ? float3(Theme.Colors.ObjectActive) : float3(Theme.Colors.ObjectSelected), 1.0f);
    }
};

using Scene = SceneT<BindlessSet>;
using SceneImageWrite = SceneT<BindlessSetImageWrite>;
using SceneImageUint = SceneT<BindlessSetImageUint>;

// The draw this instance renders, looked up through the visible-index list the cull pass produced.
// `instance_index` counts from the first instance of the batch, which is what instance_id already
// holds: Metal folds an indirect command's base instance into it.
template<typename SetT>
inline DrawData GetDrawData(const thread SceneT<SetT> &scene, uint draw_data_offset, uint instance_index) {
    const uint dense = scene.VisibleIndices(scene.View.VisibleIndexSlot)[draw_data_offset + instance_index];
    return scene.Draws(scene.View.DrawDataSlot)[dense];
}

// Normalized direction of `n`, or zero when `n` has no length.
inline float3 NormalizeOrZero(float3 n) {
    const float len = length(n);
    return len > 0.0f ? n / len : float3(0.0f);
}

#endif
