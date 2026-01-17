#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_scalar_block_layout : require

#include "SceneUBO.glsl"
#include "DrawData.glsl"

struct Vertex {
    vec3 Position;
    vec3 Normal;
};

struct WorldMatrix {
    mat4 M;
    mat4 MInv;
};

layout(set = 0, binding = 3, scalar) readonly buffer VertexBuffer {
    Vertex Vertices[];
} VertexBuffers[];

layout(set = 0, binding = 5, scalar) readonly buffer ModelBuffer {
    WorldMatrix Models[];
} ModelBuffers[];

layout(set = 0, binding = 4, scalar) readonly buffer IndexBuffer {
    uint Indices[];
} IndexBuffers[];

#ifndef BINDLESS_NO_ELEMENT_STATE
layout(set = 0, binding = 6, scalar) readonly buffer ElementStateBuffer {
    uint States[];
} ElementStateBuffers[];
#endif

layout(set = 0, binding = 7, scalar) readonly buffer ObjectIdBuffer {
    uint Ids[];
} ObjectIdBuffers[];

layout(set = 0, binding = 8, scalar) readonly buffer FaceNormalBuffer {
    vec3 Normals[];
} FaceNormalBuffers[];

layout(set = 0, binding = 9, scalar) readonly buffer DrawDataBuffer {
    DrawData Draws[];
} DrawDataBuffers[];

const uint INVALID_SLOT = 0xffffffffu;
const uint STATE_SELECTED = 1u << 0;
const uint STATE_ACTIVE = 1u << 1;

layout(push_constant) uniform PushConstants {
    uint DrawDataSlot;
    uint DrawDataOffset;
    uint Pad0;
    uint Pad1;
} pc;

DrawData GetDrawData() {
    return DrawDataBuffers[nonuniformEXT(pc.DrawDataSlot)].Draws[pc.DrawDataOffset + gl_InstanceIndex];
}
