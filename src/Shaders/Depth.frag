#version 450

layout(location = 0) out vec4 OutColor; // {depth, depth, depth, HasMesh}

void main() {
    const float depth = gl_FragCoord.z;
    OutColor = vec4(depth, depth, depth, 1);
}
