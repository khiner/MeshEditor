#version 450

layout(location = 0) in vec3 Position;
layout(location = 1) in vec3 VertexNormal;
layout(location = 2) in vec4 VertexColor;
layout(location = 3) in mat4 Model;
layout(location = 7) in mat4 InvModel; // Stores the transpose of the inverse of `Model`.

layout(location = 0) out vec3 WorldNormal; // Vertex normal transformed to world space, for lighting calculations.
layout(location = 1) out vec4 Color;
layout(location = 2) out vec3 ViewPosition;

layout(binding = 0) uniform ViewProjectionUBO {
    mat4 View;
    mat4 Proj;
} ViewProj;

void main() {
    WorldNormal = mat3(InvModel) * VertexNormal;
    Color = VertexColor;
    ViewPosition = -vec3(inverse(ViewProj.View)[3]); // Camera position in world space
    gl_Position = ViewProj.Proj * ViewProj.View * Model * vec4(Position, 1.0);
}
