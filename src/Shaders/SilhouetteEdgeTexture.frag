#version 450

layout(binding = 0) uniform sampler2D SilhouetteEdgeTexture; // Assumes each pixel of the texture contains {IsEdge, Depth, *, EdgeWeight}
layout(binding = 1) uniform SilhouetteDisplayUBO {
    vec4 Color;
} SilhouetteDisplay;

layout(location = 0) in vec2 TexCoord;
layout(location = 0) out vec4 OutColor;

void main() {
    const vec4 value = texture(SilhouetteEdgeTexture, TexCoord);
    if (value.r == 1.0) {
        const float depth = value.g;
        gl_FragDepth = depth;
        OutColor = SilhouetteDisplay.Color;
        OutColor.a *= value.a; // The alpha channel may store a normalized edge detection weight, which we use it here for alpha to support "antialiasing".
    } else {
        discard;
    }
}
