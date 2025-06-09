#version 450

layout(push_constant) uniform PC {
    uint objectId;
} pc;

layout(location = 0) out vec2 DepthObjectId; // {Depth, ObjectID)}

void main() {
    const float depth = gl_FragCoord.z;
    DepthObjectId = vec2(depth, float(pc.objectId));
}
