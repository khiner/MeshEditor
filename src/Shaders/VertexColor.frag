#version 450

layout(location = 0) in vec3 WorldNormal; // Unused
layout(location = 1) in vec4 Color;

layout(location = 0) out vec4 OutColor;

void main() {
    OutColor = Color;
}
