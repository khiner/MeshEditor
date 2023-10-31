#version 450

// Basic, single-source diffuse lighting.

layout(location = 0) in vec3 FragNormal;
layout(location = 1) in vec4 FragColor;

layout(location = 0) out vec4 OutColor;

layout(set = 0, binding = 1) uniform LightUBO {
    vec3 Direction;
    vec3 Color;
} Light;

void main() {
    float diffuse = max(dot(normalize(FragNormal), -normalize(Light.Direction)), 0.0);
    vec3 lighting = diffuse * Light.Color;
    // OutColor = vec4(FragColor.rgb * lighting, FragColor.a);
    OutColor = FragColor;
}
