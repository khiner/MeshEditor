#version 450

layout(location = 0) in vec3 InPosition;
layout(location = 1) in vec4 InColor;

layout(location = 0) out vec4 FragColor;

void main() {
    gl_Position = vec4(InPosition, 1.0);
    FragColor = InColor;
}
