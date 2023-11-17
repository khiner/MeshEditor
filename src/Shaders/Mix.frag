#version 450

layout(binding = 0) uniform sampler2D MainSceneTex;
layout(binding = 1) uniform sampler2D SilhouetteTex;
layout(binding = 3) uniform SilhouetteControlsUBO {
    float Alpha;
} SilhouetteControls;

layout(location = 0) in vec2 TexCoord;
layout(location = 0) out vec4 OutColor;

void main() {
    vec4 main_color = texture(MainSceneTex, TexCoord);
    vec4 silhouette_color = texture(SilhouetteTex, TexCoord);
    OutColor = silhouette_color.a == 1.0 ? mix(main_color, silhouette_color, SilhouetteControls.Alpha) : main_color;
}
