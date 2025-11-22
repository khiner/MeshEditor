#version 450

layout(location = 0) in vec3 Position;
layout(location = 1) in vec3 VertexNormal;
layout(location = 2) in vec4 VertexColor;
layout(location = 3) in mat4 Model;
layout(location = 7) in mat4 InvModel; // Stores the transpose of the inverse of `Model`.

layout(location = 0) out vec3 WorldNormal; // Vertex normal transformed to world space, for lighting calculations.
layout(location = 1) out vec3 WorldPosition;
layout(location = 2) out vec4 Color;

layout(binding = 0) uniform CameraUBO {
    mat4 View;
    mat4 Proj;
    vec3 Position;
} Camera;

void main() {
    WorldNormal = mat3(InvModel) * VertexNormal;
    WorldPosition = vec3(Model * vec4(Position, 1.0));
    Color = VertexColor;
    gl_Position = Camera.Proj * Camera.View * Model * vec4(Position, 1.0);
    gl_PointSize = 6.0;
}
