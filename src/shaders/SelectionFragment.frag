#version 450

#extension GL_EXT_nonuniform_qualifier : require

#include "SelectionShared.glsl"

layout(location = 1) flat in uint ObjectId;

const uint INVALID_NODE = 0xffffffffu;

void main() {
    // MoltenVK/SPIRV-Cross requires nonuniformEXT for dynamic buffer array indexing
    // when using .length() or image atomics, even when the index is uniform.
    const uint head_index = nonuniformEXT(pc.SelectionHeadImageSlot);
    const uint nodes_index = nonuniformEXT(pc.SelectionNodesSlot);
    const uint counter_index = nonuniformEXT(pc.SelectionCounterSlot);

    const uint idx = atomicAdd(Counters[counter_index].Data.Count, 1);
    if (idx >= SelectionBuffers[nodes_index].Nodes.length()) {
        atomicAdd(Counters[counter_index].Data.Overflow, 1);
        return;
    }

    SelectionBuffers[nodes_index].Nodes[idx].Depth = gl_FragCoord.z;
    SelectionBuffers[nodes_index].Nodes[idx].Id = ObjectId;

    const ivec2 coord = ivec2(gl_FragCoord.xy);
    const uint prev = imageAtomicExchange(HeadImages[head_index], coord, idx);
    SelectionBuffers[nodes_index].Nodes[idx].Next = prev;
}
