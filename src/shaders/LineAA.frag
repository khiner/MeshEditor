#version 450
#extension GL_EXT_nonuniform_qualifier : require

#include "SceneUBO.glsl"

layout(set = 0, binding = BINDING_Sampler) uniform sampler2D Samplers[];

layout(location = 0) in vec2 TexCoord;
layout(location = 0) out vec4 OutColor;

layout(push_constant) uniform PushConstants {
    uint ColorSamplerSlot;
    uint LineDataSamplerSlot;
} pc;

// Decode the perpendicular direction from encoded xy channels (perp * 0.5 + 0.5 → perp * 2 - 1)
vec2 decode_line_dir(vec2 encoded) { return encoded * 2.0 - 1.0; }

// Decode the signed perpendicular distance from encoded z channel (dist * 0.25 + 0.6 → (z - 0.6) * 4)
float decode_line_dist(float encoded) { return (encoded - 0.6) * 4.0; }

// Smooth coverage: 1.0 at the line center (|dist| <= half_width), fading to 0 over a 1px AA fringe.
float line_coverage(float dist) {
    const float half_width = ViewportTheme.EdgeWidth * 0.5;
    return 1.0 - smoothstep(half_width, half_width + 1.0, abs(dist));
}

void main() {
    const vec2 texel_size = 1.0 / SceneViewUBO.ViewportSize;

    // Check center + 4 orthogonal neighbors for line data, expand AA coverage.
    const vec2 offsets[5] = vec2[](
        vec2(0, 0),   // center
        vec2(-1, 0),  // left
        vec2( 1, 0),  // right
        vec2( 0,-1),  // down
        vec2( 0, 1)   // up
    );

    float max_coverage = 0.0;
    vec2 best_uv = TexCoord;

    for (int i = 0; i < 5; i++) {
        const vec2 n_uv = TexCoord + offsets[i] * texel_size;
        const vec4 n_line = texture(Samplers[pc.LineDataSamplerSlot], n_uv);
        if (n_line.a < 0.5) continue; // No line data at this sample

        const vec2 perp = decode_line_dir(n_line.xy);
        // Compute the distance from the CURRENT pixel to the line that passed through sample i.
        const float curr_dist = decode_line_dist(n_line.z) - dot(perp, offsets[i]);

        const float cov = line_coverage(curr_dist);
        if (cov > max_coverage) {
            max_coverage = cov;
            best_uv = n_uv;
        }
    }

    vec4 out_color = texture(Samplers[pc.ColorSamplerSlot], TexCoord);
    if (max_coverage > 0.0) {
        out_color.rgb = mix(out_color.rgb, texture(Samplers[pc.ColorSamplerSlot], best_uv).rgb, max_coverage);
    }
    OutColor = out_color;
}
