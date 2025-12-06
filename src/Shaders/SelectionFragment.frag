#version 450

layout(push_constant) uniform PC {
    uint ObjectId;
} pc;

struct SelectionNode {
    uint objectID;
    float depth;
    uint next;
};

layout(binding = 1, std430) buffer SelectionNodes {
    SelectionNode nodes[];
};

layout(binding = 2, r32ui) uniform uimage2D HeadImage;

layout(binding = 3, std430) buffer SelectionCounter {
    uint count;
    uint overflow;
};

const uint INVALID_NODE = 0xffffffffu;

void main() {
    const uint idx = atomicAdd(count, 1);
    if (idx >= nodes.length()) {
        atomicAdd(overflow, 1);
        return;
    }

    nodes[idx].objectID = pc.ObjectId;
    nodes[idx].depth = gl_FragCoord.z;

    const ivec2 coord = ivec2(gl_FragCoord.xy);
    const uint prev = imageAtomicExchange(HeadImage, coord, idx);
    nodes[idx].next = prev;
}
