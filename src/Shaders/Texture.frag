#version 450

layout(binding = 0) uniform sampler2D Texture;

layout(location = 0) in vec2 TexCoord;
layout(location = 0) out vec4 OutColor;

void main() {
    OutColor = texture(Texture, TexCoord);
}
