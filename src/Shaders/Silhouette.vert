#version 450

layout(location = 0) in vec3 Position;

layout(set = 0, binding = 0) uniform TransformUBO {
    mat4 Model;
    mat4 View;
    mat4 Projection;
} Transform;

void main() {
    gl_Position = Transform.Projection * Transform.View * Transform.Model * vec4(Position, 1.0);
}
