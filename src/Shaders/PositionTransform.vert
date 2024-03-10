#version 450

layout(binding = 0) uniform TransformUBO {
    mat4 View;
    mat4 Projection;
} Transform;

layout(location = 0) in vec3 Position;
layout(location = 3) in mat4 Model;

void main() {
    gl_Position = Transform.Projection * Transform.View * Model * vec4(Position, 1.0);
}
