#version 450

layout(binding = 0) uniform CameraUBO {
    mat4 View;
    mat4 Proj;
    vec3 Position;
} Camera;

layout(location = 0) in vec3 Position;
layout(location = 3) in mat4 Model;

void main() {
    gl_Position = Camera.Proj * Camera.View * Model * vec4(Position, 1.0);
}
