#ifndef SELECTION_SHARED_GLSL
#define SELECTION_SHARED_GLSL

#extension GL_EXT_scalar_block_layout : require

#ifndef SELECTION_CUSTOM_DRAW_PASS_PC
#include "DrawPassPushConstants.glsl"
#endif
#include "SelectionCounters.glsl"
#include "SelectionNode.glsl"

layout(set = 0, binding = BINDING_Image, r32ui) uniform uimage2D HeadImages[];

layout(set = 0, binding = BINDING_Buffer, scalar) buffer SelectionNodes {
    SelectionNode Nodes[];
} SelectionBuffers[];

layout(set = 0, binding = BINDING_Buffer, scalar) buffer SelectionCounter {
    SelectionCounters Data;
} Counters[];

#endif
