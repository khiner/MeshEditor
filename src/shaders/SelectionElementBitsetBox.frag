#version 450

#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_scalar_block_layout : require

#define SELECTION_CUSTOM_DRAW_PASS_PC 1
#include "SelectionElementPushConstants.glsl"

layout(location = 0) flat in uint ElementId;

layout(early_fragment_tests) in;

layout(set = 0, binding = BINDING_Buffer, scalar) buffer BoxSelectResult {
    uint Bits[];
} BoxResults[];

void main() {
    if (ElementId == 0u) return;
    const uvec2 frag_px = uvec2(gl_FragCoord.xy);
    if (frag_px.x < pc.BoxMinX || frag_px.x > pc.BoxMaxX ||
        frag_px.y < pc.BoxMinY || frag_px.y > pc.BoxMaxY) return;
    const uint bit_idx = ElementId - 1u;
    // MoltenVK/SPIRV-Cross requires nonuniformEXT for dynamic buffer array indexing.
    atomicOr(BoxResults[nonuniformEXT(pc.BoxResultSlot)].Bits[bit_idx >> 5u], 1u << (bit_idx & 31u));
}
