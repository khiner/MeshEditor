#version 450

layout(push_constant) uniform PC {
    uint objectIndex; // [0, N)
} pc;

layout(location = 0) out vec2 OutColor; // {depth, ObjectID=(ObjectIndex + 1)}

void main() {
    const float depth = gl_FragCoord.z;
    OutColor = vec2(depth, float(pc.objectIndex + 1));
}
