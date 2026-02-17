#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

#include "SceneUBO.glsl"
#include "DrawData.glsl"
#include "DrawPassPushConstants.glsl"
#include "Vertex.glsl"
#include "BoneDeformVertex.glsl"
#include "MorphTargetVertex.glsl"
#include "PunctualLight.glsl"
#include "WorldTransform.glsl"
#include "TRSUtils.glsl"

layout(set = 0, binding = BINDING_VertexBuffer, scalar) readonly buffer VertexBuffer {
    Vertex Vertices[];
} VertexBuffers[];

layout(set = 0, binding = BINDING_ModelBuffer, scalar) readonly buffer ModelBuffer {
    WorldTransform Models[];
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

const uint INVALID_SLOT = 0xffffffffu;
const uint STATE_SELECTED = 1u << 0;
const uint STATE_ACTIVE = 1u << 1;

DrawData GetDrawData() {
    return DrawDataBuffers[nonuniformEXT(pc.DrawData.Slot)].Draws[pc.DrawData.Offset + gl_InstanceIndex];
}
