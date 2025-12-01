#version 450

layout(push_constant) uniform PC {
    uint ObjectId;
} pc;

// Note: Using separate uint fields instead of uvec2 since std430 seems to automatically reorder for alignment when using uvec2.
struct SelectionFragment {
    uint objectID;
    uint pixel_x, pixel_y;
    float depth;
};

#define MAX_SELECTION_FRAGMENTS 4000000

layout(binding = 1, std430) buffer SelectionBuffer {
    uint fragment_count;
    SelectionFragment fragments[];
};

void main() {
    uint idx = atomicAdd(fragment_count, 1);
    if (idx >= MAX_SELECTION_FRAGMENTS) return;

    fragments[idx].objectID = pc.ObjectId;
    fragments[idx].pixel_x = uint(gl_FragCoord.x);
    fragments[idx].pixel_y = uint(gl_FragCoord.y);
    fragments[idx].depth = gl_FragCoord.z;
}
