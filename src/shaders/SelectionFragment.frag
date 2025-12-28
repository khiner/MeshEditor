#version 450
#extension GL_EXT_nonuniform_qualifier : require

#define BINDLESS_NO_ELEMENT_STATE 1
#include "Bindless.glsl"

struct SelectionNode {
    uvec2 Pixel;
    float Depth;
    uint ObjectId;
    uint Next;
};

layout(set = 0, binding = 1, r32ui) uniform uimage2D HeadImages[];

layout(set = 0, binding = 6, std430) buffer SelectionNodes {
    SelectionNode Nodes[];
} SelectionBuffers[];

layout(set = 0, binding = 6, std430) buffer SelectionCounter {
    uint Count;
    uint Overflow;
} Counters[];

layout(location = 0) flat in uint InstanceIndex;

const uint INVALID_NODE = 0xffffffffu;

layout(early_fragment_tests) in;

void main() {
    // MoltenVK/SPIRV-Cross requires nonuniformEXT for dynamic buffer array indexing
    // when using .length() or image atomics, even when the index is uniform.
    const uint head_index = nonuniformEXT(pc.VertexCountOrHeadImageSlot);
    const uint nodes_index = nonuniformEXT(pc.SelectionNodesSlot);
    const uint counter_index = nonuniformEXT(pc.SelectionCounterSlot);

    const uint idx = atomicAdd(Counters[counter_index].Count, 1);
    if (idx >= SelectionBuffers[nodes_index].Nodes.length()) {
        atomicAdd(Counters[counter_index].Overflow, 1);
        return;
    }

    const uint objectId = ObjectIdBuffers[pc.ObjectIdSlot].Ids[pc.FirstInstance + InstanceIndex];
    SelectionBuffers[nodes_index].Nodes[idx].Pixel = uvec2(gl_FragCoord.xy);
    SelectionBuffers[nodes_index].Nodes[idx].Depth = gl_FragCoord.z;
    SelectionBuffers[nodes_index].Nodes[idx].ObjectId = objectId;

    const ivec2 coord = ivec2(gl_FragCoord.xy);
    const uint prev = imageAtomicExchange(HeadImages[head_index], coord, idx);
    SelectionBuffers[nodes_index].Nodes[idx].Next = prev;
}
