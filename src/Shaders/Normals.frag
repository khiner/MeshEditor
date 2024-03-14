#version 450

layout(location = 0) in vec3 WorldNormal;

layout(location = 0) out vec4 OutColor;

void main() {
    const vec3 normal_color = normalize(WorldNormal) * 0.5 + 0.5; // Map from [-1, 1] to [0, 1].
    OutColor = vec4(normal_color, 1.0);
}
