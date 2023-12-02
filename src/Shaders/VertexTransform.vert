#version 450

layout(location = 0) in vec3 Position;
layout(location = 1) in vec3 VertexNormal;
layout(location = 2) in vec4 VertexColor;

layout(location = 0) out vec3 FragNormal;
layout(location = 1) out vec4 FragColor;
layout(location = 2) out vec3 FragViewPosition;

layout(set = 0, binding = 0) uniform TransformUBO {
    mat4 Model;
    mat4 View;
    mat4 Projection;
} Transform;

void main() {
    FragNormal = VertexNormal;
    FragColor = VertexColor;
    FragViewPosition = -vec3(inverse(Transform.View)[3]); // camera position
    gl_Position = Transform.Projection * Transform.View * Transform.Model * vec4(Position, 1.0);
}
