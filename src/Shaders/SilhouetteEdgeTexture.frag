#version 450

layout(binding = 0) uniform sampler2D SilhouetteEdgeTexture; // Assumes {Depth, ObjectID}
layout(binding = 1) uniform SilhouetteDisplayUBO {
    vec4 Color;
} SilhouetteDisplay;

layout(location = 0) in vec2 TexCoord;
layout(location = 0) out vec4 OutColor;

void main() {
    const vec2 value = texture(SilhouetteEdgeTexture, TexCoord).xy; // {Depth, ObjectID}
    if (value.y > 0.5) {
        gl_FragDepth = value.x;
        OutColor = SilhouetteDisplay.Color;
    } else {
        discard;
    }
}
