#version 450

layout(binding = 0) uniform sampler2D SilhouetteEdgeSampler; // Assumes {Depth, ObjectID}
layout(binding = 1) uniform SilhouetteDisplayUBO {
    vec4 ActiveColor;
    vec4 SelectedColor;
} SilhouetteDisplay;

layout(location = 0) in vec2 TexCoord;
layout(location = 0) out vec4 OutColor;

void main() {
    const vec2 value = texture(SilhouetteEdgeSampler, TexCoord).xy; // {Depth, ObjectID}
    if (value.y < 1) {
        discard;
    } else {
        gl_FragDepth = value.x;
        if (value.y == 1) OutColor = SilhouetteDisplay.ActiveColor;
        else OutColor = SilhouetteDisplay.SelectedColor;
    }
}
