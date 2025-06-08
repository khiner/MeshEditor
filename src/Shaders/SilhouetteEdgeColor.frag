#version 450

layout(binding = 0) uniform sampler2D ObjectIdSampler;
layout(binding = 1) uniform SilhouetteDisplayUBO {
    vec4 ActiveColor;
    vec4 SelectedColor;
} SilhouetteDisplay;

layout(location = 0) in vec2 TexCoord;
layout(location = 0) out vec4 OutColor;

void main() {
    const float object_id = texture(ObjectIdSampler, TexCoord).x;
    if (object_id < 1) {
        discard;
    } else {
        if (object_id == 1) OutColor = SilhouetteDisplay.ActiveColor;
        else OutColor = SilhouetteDisplay.SelectedColor;
    }
}
