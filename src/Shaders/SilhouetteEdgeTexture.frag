#version 450

layout(binding = 0) uniform sampler2D SilhouetteEdgeTexture;
layout(binding = 1) uniform SilhouetteDisplayUBO {
    vec4 Color;
} SilhouetteDisplay;

layout(location = 0) in vec2 TexCoord;
layout(location = 0) out vec4 OutColor;

void main() {
    const vec4 value = texture(SilhouetteEdgeTexture, TexCoord);
    if (value.r == 1.0) { // The r channel is 1 if the pixel is on the edge.
        const float depth = value.g; // g and b channels both store the depth of the geometry's detected edge.
        const float alpha = min(value.a * 2, 1); // The alpha channel stores a normalized version of the edge detection amount, and we use it here for alpha smoothing.
        gl_FragDepth = depth;
        OutColor = SilhouetteDisplay.Color;
        OutColor.a *= alpha;
    } else {
        discard;
    }
}
