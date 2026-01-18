#ifndef SELECTION_SHARED_GLSL
#define SELECTION_SHARED_GLSL

#extension GL_EXT_scalar_block_layout : require

#include "BindlessBindings.glsl"

struct SelectionNode {
    float Depth;
    uint ObjectId;
    uint Next;
    uint Padding0;
};

layout(set = 0, binding = BINDING_Image, r32ui) uniform uimage2D HeadImages[];

layout(set = 0, binding = BINDING_Buffer, scalar) buffer SelectionNodes {
    SelectionNode Nodes[];
} SelectionBuffers[];

layout(set = 0, binding = BINDING_Buffer, scalar) buffer SelectionCounter {
    uint Count;
    uint Overflow;
} Counters[];

layout(push_constant) uniform SelectionPushConstants {
    uint DrawDataSlot;
    uint DrawDataOffset;
    uint SelectionHeadImageSlot;
    uint SelectionNodesSlot;
    uint SelectionCounterSlot;
    uint Pad0;
    uint Pad1;
    uint Pad2;
} pc;

#endif
