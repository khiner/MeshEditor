#version 450

#extension GL_EXT_nonuniform_qualifier : require

#define SELECTION_CUSTOM_DRAW_PASS_PC 1
#include "SelectionElementPushConstants.glsl"
#include "SelectionShared.glsl"
#include "BindlessBindings.glsl"

layout(location = 0) flat in uint ElementId;

const uint INVALID_NODE = 0xffffffffu;
const uint SELECTION_ELEMENT_OUTPUT_BITSET = 1u << 0u;
const uint SELECTION_ELEMENT_CLIP_TO_BOX = 1u << 1u;

layout(early_fragment_tests) in;

layout(set = 0, binding = BINDING_Buffer, scalar) buffer BoxSelectResult {
    uint Bits[];
} BoxResults[];

void WriteLinkedList() {
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
    SelectionBuffers[nodes_index].Nodes[idx].Id = ElementId;

    const ivec2 coord = ivec2(gl_FragCoord.xy);
    const uint prev = imageAtomicExchange(HeadImages[head_index], coord, idx);
    SelectionBuffers[nodes_index].Nodes[idx].Next = prev;
}

void WriteBitset() {
    if (ElementId == 0u) return;
    if ((pc.Flags & SELECTION_ELEMENT_CLIP_TO_BOX) != 0u) {
        const uvec2 frag_px = uvec2(gl_FragCoord.xy);
        if (frag_px.x < pc.BoxMinX || frag_px.x > pc.BoxMaxX ||
            frag_px.y < pc.BoxMinY || frag_px.y > pc.BoxMaxY) return;
    }
    const uint bit_idx = ElementId - 1u;
    // MoltenVK/SPIRV-Cross requires nonuniformEXT for dynamic buffer array indexing.
    atomicOr(BoxResults[nonuniformEXT(pc.BoxResultSlot)].Bits[bit_idx >> 5u], 1u << (bit_idx & 31u));
}

void main() {
    if ((pc.Flags & SELECTION_ELEMENT_OUTPUT_BITSET) != 0u) {
        WriteBitset();
    } else {
        WriteLinkedList();
    }
}
