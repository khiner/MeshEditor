#version 450

layout(location = 0) in vec3 InPosition;
layout(location = 1) in vec3 InNormal;
layout(location = 2) in vec4 InColor;

layout(location = 0) out vec3 FragNormal;
layout(location = 1) out vec4 FragColor;

layout(set = 0, binding = 0) uniform TransformUBO {
    mat4 Model;
    mat4 View;
    mat4 Projection;
} Transform;

void main() {
    gl_Position = Transform.Projection * Transform.View * Transform.Model * vec4(InPosition, 1.0);
    FragNormal = InNormal;
    FragColor = InColor;
}
