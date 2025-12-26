// Shared bindless buffer definitions
// Include after declaring #version and extensions

#extension GL_EXT_scalar_block_layout : require

#include "SceneUBO.glsl"

struct Vertex {
    vec3 Position;
    vec3 Normal;
    vec4 Color;
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

layout(set = 0, binding = 7, scalar) readonly buffer ObjectIdBuffer {
    uint Ids[];
} ObjectIdBuffers[];

const uint INVALID_SLOT = 0xffffffffu;

layout(push_constant) uniform PushConstants {
    uint VertexSlot;
    uint IndexSlot;
    uint ModelSlot;
    uint FirstInstance;
    uint ObjectIdSlot; // Slot for per-instance ObjectId buffer (INVALID_SLOT if unused)
    uint VertexCountOrHeadImageSlot; // HeadImageSlot for selection fragment
    uint SelectionNodesSlot;
    uint SelectionCounterSlot;
    uint ElementIdOffset;
} pc;
