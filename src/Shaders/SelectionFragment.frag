#version 450
#extension GL_EXT_nonuniform_qualifier : require

#include "Bindless.glsl"

struct SelectionNode {
    uint objectID;
    float depth;
    uint next;
};

layout(set = 0, binding = 1, r32ui) uniform uimage2D HeadImages[];

layout(set = 0, binding = 6, std430) buffer SelectionNodes {
    SelectionNode nodes[];
} selectionBuffers[];

layout(set = 0, binding = 6, std430) buffer SelectionCounter {
    uint count;
    uint overflow;
} counters[];

const uint INVALID_NODE = 0xffffffffu;

void main() {
    const uint head_index = nonuniformEXT(pc.VertexCountOrHeadImageSlot);
    const uint nodes_index = nonuniformEXT(pc.SelectionNodesSlot);
    const uint counter_index = nonuniformEXT(pc.SelectionCounterSlot);

    const uint idx = atomicAdd(counters[counter_index].count, 1);
    if (idx >= selectionBuffers[nodes_index].nodes.length()) {
        atomicAdd(counters[counter_index].overflow, 1);
        return;
    }

    selectionBuffers[nodes_index].nodes[idx].objectID = pc.ObjectId;
    selectionBuffers[nodes_index].nodes[idx].depth = gl_FragCoord.z;

    const ivec2 coord = ivec2(gl_FragCoord.xy);
    const uint prev = imageAtomicExchange(HeadImages[head_index], coord, idx);
    selectionBuffers[nodes_index].nodes[idx].next = prev;
}
