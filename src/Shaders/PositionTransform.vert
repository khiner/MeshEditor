#version 450

layout(binding = 0) uniform ViewProjectionUBO {
    mat4 View;
    mat4 Proj;
} ViewProj;

layout(location = 0) in vec3 Position;
layout(location = 3) in mat4 Model;

void main() {
    gl_Position = ViewProj.Proj * ViewProj.View * Model * vec4(Position, 1.0);
}
