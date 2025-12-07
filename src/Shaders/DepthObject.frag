#version 450

layout(push_constant) uniform PC {
    uint VertexSlot;
    uint IndexSlot;
    uint ModelSlot;
    uint FirstIndex;
    uint VertexOffset;
    uint FirstInstance;
    uint ObjectId;
    uint _reserved0;
    uint _reserved1;
    uint _reserved2;
} pc;

layout(location = 0) out vec2 DepthObjectId; // {Depth, ObjectID}

void main() {
    DepthObjectId = vec2(gl_FragCoord.z, float(pc.ObjectId));
}
