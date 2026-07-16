#version 450

layout(location = 0) in vec4 Motion;
layout(location = 0) out vec4 OutMotion;

// Pixels no geometry covers keep the target's zero clear, which reads as static.
void main() {
    OutMotion = Motion;
}
