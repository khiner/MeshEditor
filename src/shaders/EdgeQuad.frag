#version 450

#include "SceneUBO.glsl"

layout(location = 0) noperspective in float EdgeCoord;
layout(location = 1) in vec4 Color;

layout(location = 0) out vec4 OutColor;
layout(location = 1) out vec4 OutLineData;

void main() {
    // Matches Blender's overlay_shader_shared.hh constant values
    const float DISC_RADIUS = 0.5641895835477563 * 1.05; // M_1_SQRTPI * 1.05
    const float LINE_SMOOTH_START = 0.5 - DISC_RADIUS;
    const float LINE_SMOOTH_END = 0.5 + DISC_RADIUS;

    // Distance from edge core boundary.
    float dist = abs(EdgeCoord) - max(ViewportTheme.EdgeWidth - 0.5, 0.0);
    float mix_w = smoothstep(LINE_SMOOTH_START, LINE_SMOOTH_END, dist);

    OutColor = vec4(Color.rgb, Color.a * (1.0 - mix_w));
    OutLineData = vec4(0.0); // Opt out of LineAA composite (edge quads handle their own AA).
}
