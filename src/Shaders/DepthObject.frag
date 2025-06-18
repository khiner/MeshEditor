#version 450

layout(push_constant) uniform PC {
    uint ObjectId;
} pc;

layout(location = 0) out vec2 DepthObjectId; // {Depth, ObjectID}

void main() {
    DepthObjectId = vec2(gl_FragCoord.z, float(pc.ObjectId));
}
