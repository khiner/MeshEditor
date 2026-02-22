#ifndef SELECTION_TRAVERSAL_GLSL
#define SELECTION_TRAVERSAL_GLSL

#include "BindlessBindings.glsl"
#include "SelectionNode.glsl"

const uint INVALID_SELECTION_NODE = 0xffffffffu;

layout(set = 0, binding = BINDING_Image, r32ui) uniform uimage2D HeadImages[];

layout(set = 0, binding = BINDING_Buffer, scalar) buffer SelectionNodes {
    SelectionNode Nodes[];
} SelectionBuffers[];

ivec2 SelectionImageSize(uint head_image_index) {
    return imageSize(HeadImages[nonuniformEXT(head_image_index)]);
}

bool SelectionPixelInBounds(ivec2 pixel, ivec2 size) {
    return pixel.x >= 0 && pixel.y >= 0 && pixel.x < size.x && pixel.y < size.y;
}

uint SelectionHeadAt(uint head_image_index, ivec2 pixel) {
    return imageLoad(HeadImages[nonuniformEXT(head_image_index)], pixel).r;
}

SelectionNode SelectionNodeAt(uint selection_nodes_index, uint node_idx) {
    return SelectionBuffers[nonuniformEXT(selection_nodes_index)].Nodes[node_idx];
}

bool SelectionIdInRange(uint id, uint max_id) {
    return id > 0u && id <= max_id;
}

#endif
