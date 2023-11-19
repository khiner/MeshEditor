#version 450

// Basic, single diffuce light source with ambient lighting.

layout(location = 0) in vec3 FragNormal;
layout(location = 1) in vec4 FragColor;

layout(location = 0) out vec4 OutColor;

layout(set = 0, binding = 1) uniform LightUBO {
    vec4 ColorAndAmbient;
    vec3 Direction;
} Light;

void main() {
    const vec3 light_color = Light.ColorAndAmbient.xyz;
    const float light_ambient = Light.ColorAndAmbient.w;
    const vec3 diffuse_lighting = max(dot(normalize(FragNormal), -normalize(Light.Direction)), 0.0) * light_color;
    const vec3 ambient_lighting = light_color * light_ambient;
    const vec3 lighting = diffuse_lighting + ambient_lighting;

    OutColor = vec4(FragColor.rgb * lighting, FragColor.a);
}
