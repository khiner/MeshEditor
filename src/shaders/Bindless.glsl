#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

#include "SceneUBO.glsl"
#include "DrawData.glsl"
#ifndef BINDLESS_CUSTOM_DRAW_PASS_PC
#include "MainDrawPushConstants.glsl"
#endif
#include "Vertex.glsl"
#include "BoneDeformVertex.glsl"
#include "MorphTargetVertex.glsl"
#include "PBRMaterial.glsl"
#include "PunctualLight.glsl"
#include "Transform.glsl"
#include "TRSUtils.glsl"

layout(set = 0, binding = BINDING_VertexBuffer, scalar) readonly buffer VertexBuffer {
    Vertex Vertices[];
} VertexBuffers[];

layout(set = 0, binding = BINDING_ModelBuffer, scalar) readonly buffer ModelBuffer {
    Transform Models[];
} ModelBuffers[];

layout(set = 0, binding = BINDING_IndexBuffer, scalar) readonly buffer IndexBuffer {
    uint Indices[];
} IndexBuffers[];

#ifndef BINDLESS_NO_ELEMENT_STATE
layout(set = 0, binding = BINDING_Buffer, scalar) readonly buffer ElementStateBuffer {
    uint8_t States[];
} ElementStateBuffers[];
#endif

layout(set = 0, binding = BINDING_ObjectIdBuffer, scalar) readonly buffer ObjectIdBuffer {
    uint Ids[];
} ObjectIdBuffers[];

layout(set = 0, binding = BINDING_DrawDataBuffer, scalar) readonly buffer DrawDataBuffer {
    DrawData Draws[];
} DrawDataBuffers[];

layout(set = 0, binding = BINDING_Buffer, scalar) readonly buffer VisibleIndexBuffer {
    uint Indices[];
} VisibleIndexBuffers[];

layout(set = 0, binding = BINDING_InstanceStateBuffer, scalar) readonly buffer InstanceStateBuffer {
    uint8_t States[];
} InstanceStateBuffers[];

layout(set = 0, binding = BINDING_BoneDeformBuffer, scalar) readonly buffer BoneDeformBuffer {
    BoneDeformVertex Vertices[];
} BoneDeformBuffers[];

layout(set = 0, binding = BINDING_ArmatureDeformBuffer, scalar) readonly buffer ArmatureDeformBuffer {
    mat4 Matrices[];
} ArmatureDeformBuffers[];

layout(set = 0, binding = BINDING_MorphTargetBuffer, scalar) readonly buffer MorphTargetBuffer {
    MorphTargetVertex Vertices[];
} MorphTargetBuffers[];

layout(set = 0, binding = BINDING_MorphWeightBuffer, scalar) readonly buffer MorphWeightBuffer {
    float Weights[];
} MorphWeightBuffers[];

layout(set = 0, binding = BINDING_VertexClassBuffer, scalar) readonly buffer VertexClassBuffer {
    uint8_t Classes[];
} VertexClassBuffers[];

layout(set = 0, binding = BINDING_LightBuffer, scalar) readonly buffer LightBufferBlock {
    PunctualLight Lights[];
} LightBuffers[];

layout(set = 0, binding = BINDING_MaterialBuffer, scalar) readonly buffer MaterialBufferBlock {
    PBRMaterial Materials[];
} MaterialBuffers[];

layout(set = 0, binding = BINDING_PrimitiveMaterialBuffer, scalar) readonly buffer PrimitiveMaterialBufferBlock {
    uint MaterialIndices[];
} PrimitiveMaterialBuffers[];

layout(set = 0, binding = BINDING_FacePrimitiveBuffer, scalar) readonly buffer FacePrimitiveBufferBlock {
    uint PrimitiveIndices[];
} FacePrimitiveBuffers[];

layout(set = 0, binding = BINDING_CornerTangentBuffer, scalar) readonly buffer CornerTangentBufferBlock {
    vec4 Tangents[];
} CornerTangentBuffers[];

layout(set = 0, binding = BINDING_CornerColorBuffer, scalar) readonly buffer CornerColorBufferBlock {
    vec4 Colors[];
} CornerColorBuffers[];

layout(set = 0, binding = BINDING_CornerUvBuffer, scalar) readonly buffer CornerUvBufferBlock {
    vec2 Uvs[];
} CornerUvBuffers[];

layout(set = 0, binding = BINDING_Buffer, scalar) readonly buffer CornerClassBufferBlock {
    uint Classes[];
} CornerClassBuffers[];

// Authored corner-normal (polar, azimuth) offsets from the derived normal.
// They pack to the corners the mask marks present.
layout(set = 0, binding = BINDING_Buffer, scalar) readonly buffer CustomCornerNormalBufferBlock {
    vec2 Offsets[];
} CustomCornerNormalBuffers[];

// Custom corner-normal presence: one (bitset word, exclusive rank) pair per 32 mesh corners.
layout(set = 0, binding = BINDING_Buffer, scalar) readonly buffer CustomCornerMaskBufferBlock {
    uvec2 WordRanks[];
} CustomCornerMaskBuffers[];

// Composed sector normal per seam corner, for static face draws.
layout(set = 0, binding = BINDING_Buffer, scalar) readonly buffer BaseSeamNormalBufferBlock {
    vec3 Normals[];
} BaseSeamNormalBuffers[];

// Base per-vertex normals, one vec3 per vertex-arena slot.
// Derived smooth normals for triangle meshes, authored normals for face-less meshes (bones, point clouds).
layout(set = 0, binding = BINDING_Buffer, scalar) readonly buffer BaseVertexNormalBufferBlock {
    vec3 Normals[];
} BaseVertexNormalBuffers[];

// Derived face normals, one vec3 per face-arena slot, for static face draws.
layout(set = 0, binding = BINDING_Buffer, scalar) readonly buffer BaseFaceNormalBufferBlock {
    vec3 Normals[];
} BaseFaceNormalBuffers[];

// Compute-pass tiling: one element per workgroup, (entry index, tile index) over 256-element tiles.
layout(set = 0, binding = BINDING_Buffer, scalar) readonly buffer TileMapBufferBlock {
    uvec2 Tiles[];
} TileMapBuffers[];

// Current-pose vertex positions in mesh-local space, and the normals derived from them.
layout(set = 0, binding = BINDING_Buffer, scalar) readonly buffer PosedPositionBufferBlock {
    vec3 Positions[];
} PosedPositionBuffers[];

layout(set = 0, binding = BINDING_Buffer, scalar) readonly buffer PosedVertexNormalBufferBlock {
    vec3 Normals[];
} PosedVertexNormalBuffers[];

layout(set = 0, binding = BINDING_Buffer, scalar) readonly buffer PosedSeamNormalBufferBlock {
    vec3 Normals[];
} PosedSeamNormalBuffers[];

layout(set = 0, binding = BINDING_Buffer, scalar) readonly buffer PosedFaceNormalBufferBlock {
    vec3 Normals[];
} PosedFaceNormalBuffers[];

// Weight-summed authored morph normal deltas, indexed like the posed positions.
layout(set = 0, binding = BINDING_Buffer, scalar) readonly buffer PosedMorphNormalDeltaBufferBlock {
    vec3 Deltas[];
} PosedMorphNormalDeltaBuffers[];

const uint INVALID_SLOT = 0xffffffffu;
const uint INVALID_OFFSET = 0xffffffffu;
const uint STATE_SELECTED = 1u << 0;
const uint STATE_ACTIVE = 1u << 1;
const uint STATE_EXCITED = 1u << 2;

#ifndef BINDLESS_NO_DRAW_LOOKUP
DrawData GetDrawData() {
    const uint dense = VisibleIndexBuffers[nonuniformEXT(SceneViewUBO.VisibleIndexSlot)].Indices[pc.DrawDataOffset + gl_InstanceIndex];
    return DrawDataBuffers[nonuniformEXT(SceneViewUBO.DrawDataSlot)].Draws[dense];
}
#endif

// Mesh-local vertex position: the pose pre-pass's current-pose position when the draw has one.
vec3 GetLocalPosition(DrawData draw, uint idx) {
    return draw.PosedPositionOffset != INVALID_OFFSET ?
        PosedPositionBuffers[nonuniformEXT(SceneViewUBO.PosedPositionSlot)].Positions[draw.PosedPositionOffset + idx] :
        VertexBuffers[draw.VertexSlot].Vertices[draw.VertexOffset + idx].Position;
}

// Per-vertex normal: the posed normal when the draw has one, else the base normal at the vertex-arena slot.
vec3 GetVertexNormal(DrawData draw, uint idx) {
    return draw.PosedVertexNormalOffset != INVALID_OFFSET ?
        PosedVertexNormalBuffers[nonuniformEXT(SceneViewUBO.PosedVertexNormalSlot)].Normals[draw.PosedVertexNormalOffset + idx] :
        BaseVertexNormalBuffers[nonuniformEXT(SceneViewUBO.BaseVertexNormalSlot)].Normals[draw.VertexOffset + idx];
}

// Normalized direction of `n`, or zero when `n` has no length.
vec3 NormalizeOrZero(vec3 n) {
    const float len = length(n);
    return len > 0.0 ? n / len : vec3(0);
}
