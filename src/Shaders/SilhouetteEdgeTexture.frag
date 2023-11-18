#version 450

layout(binding = 0) uniform sampler2D SilhouetteEdgeTexture;
layout(binding = 1) uniform SilhouetteDisplayUBO {
    vec4 Color;
} SilhouetteDisplay;

layout(location = 0) in vec2 TexCoord;
layout(location = 0) out vec4 OutColor;

void main() {
    const vec4 value = texture(SilhouetteEdgeTexture, TexCoord);
    const bool on = value.g == 1.0;
    if (on) {
        const float depth = value.r;
        const float alpha = value.a; // The edge detection has some built-in alpha smoothing.
        gl_FragDepth = depth;
        OutColor = SilhouetteDisplay.Color;
        OutColor.a *= alpha;
    } else {
        discard;
    }
}
