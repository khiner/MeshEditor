#version 450

layout(binding = 0) uniform sampler2D ObjectIdSampler;
layout(binding = 1) uniform SilhouetteDisplayUBO {
    vec4 ActiveColor;
    vec4 SelectedColor;
} SilhouetteDisplay;

layout(location = 0) in vec2 TexCoord;
layout(location = 0) out vec4 EdgeColor;

void main() {
    const ivec2 texel = ivec2(TexCoord * textureSize(ObjectIdSampler, 0));
    const float object_id  = texelFetch(ObjectIdSampler, texel, 0).r;
    if (object_id < 1) {
        discard;
    } else {
        EdgeColor = object_id == 1 ? SilhouetteDisplay.ActiveColor : SilhouetteDisplay.SelectedColor;
    }
}
