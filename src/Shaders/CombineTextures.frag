#version 450

layout(binding = 0) uniform sampler2D Tex1;
layout(binding = 1) uniform sampler2D Tex2;

layout(location = 0) in vec2 TexCoord;
layout(location = 0) out vec4 OutColor;

void main() {
    vec4 tex1_color = texture(Tex1, TexCoord);
    vec4 tex2_color = texture(Tex2, TexCoord);
    OutColor = vec4(mix(tex1_color.rgb, tex2_color.rgb, tex2_color.a), 1);
}
