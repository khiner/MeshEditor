#ifndef SELECTION_SHARED_GLSL
#define SELECTION_SHARED_GLSL

#extension GL_EXT_scalar_block_layout : require

#include "BindlessBindings.glsl"
#include "DrawPassPushConstants.glsl"
#include "SelectionCounters.glsl"
#include "SelectionNode.glsl"

layout(set = 0, binding = BINDING_Image, r32ui) uniform uimage2D HeadImages[];

layout(set = 0, binding = BINDING_Buffer, scalar) buffer SelectionNodes {
    SelectionNode Nodes[];
} SelectionBuffers[];

layout(set = 0, binding = BINDING_Buffer, scalar) buffer SelectionCounter {
    SelectionCounters Data;
} Counters[];

layout(push_constant) uniform SelectionPushConstants {
    DrawPassPushConstants Data;
} pc;

#endif
