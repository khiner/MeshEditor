#version 450
#extension GL_EXT_nonuniform_qualifier : require

#include "SceneUBO.glsl"
#include "tonemapping.glsl"

layout(set = 0, binding = BINDING_Sampler) uniform sampler2D Samplers[];

layout(location = 0) in vec2 TexCoord;
layout(location = 0) out vec4 OutColor;

layout(push_constant) uniform PushConstants {
    uint SceneColorSamplerSlot;
    uint OverlayColorSamplerSlot;
    uint LineDataSamplerSlot;
    // The active view transform: 0 encodes only, 1 tone maps first, 2 passes debug values through.
    uint ViewTransform;
    vec4 Backdrop; // Display-referred viewport background.
} pc;

// Decode the perpendicular direction from encoded xy channels
vec2 decode_line_dir(vec2 encoded) { return encoded * 2.0 - 1.0; }

// Decode the signed perpendicular distance from encoded z channel
float decode_line_dist(float encoded) { return (encoded - 0.6) * 4.0; }

// Blender AA constants (overlay_shader_shared.hh).
const float DISC_RADIUS = 0.5641895835477563 * 1.05; // M_1_SQRTPI * 1.05
const float LINE_SMOOTH_START = 0.5 - DISC_RADIUS;
const float LINE_SMOOTH_END = 0.5 + DISC_RADIUS;

void main() {
    const vec2 texel_size = 1.0 / SceneViewUBO.ViewportSize;

    // Check center + 4 orthogonal neighbors for line data, expand AA coverage.
    const vec2 offsets[5] = vec2[](
        vec2(0, 0),  // center
        vec2(-1, 0), // left
        vec2( 1, 0), // right
        vec2( 0,-1), // down
        vec2( 0, 1)  // up
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
        const float cov = smoothstep(LINE_SMOOTH_END, LINE_SMOOTH_START, abs(curr_dist) - (ViewportTheme.EdgeWidth - 0.5));
        if (cov > max_coverage) {
            max_coverage = cov;
            best_uv = n_uv;
        }
    }

    // The overlay layer is premultiplied, so alpha interpolates along with color: borrowing a
    // neighbor's line at partial coverage yields that line's color at that coverage.
    vec4 overlay = texture(Samplers[pc.OverlayColorSamplerSlot], TexCoord);
    if (max_coverage > 0.0) {
        overlay = mix(overlay, texture(Samplers[pc.OverlayColorSamplerSlot], best_uv), max_coverage);
    }

    // The scene layer is premultiplied. Recover radiance for the view transform, then re-apply
    // coverage, so partial-coverage pixels keep their weight.
    const vec4 scene = texture(Samplers[pc.SceneColorSamplerSlot], TexCoord);
    const vec3 radiance = scene.a > 0.0 ? scene.rgb / scene.a : vec3(0.0);
    const vec3 scene_display = pc.ViewTransform == 2u ? radiance :
        pc.ViewTransform == 1u ? linearToDisplay(radiance) : linearTosRGB(radiance);
    // The backdrop is UI. It sits under the scene in display space, and the view transform skips it.
    const vec3 base = scene_display * scene.a + pc.Backdrop.rgb * (1.0 - scene.a);
    OutColor = vec4(base * (1.0 - overlay.a) + overlay.rgb, 1.0);
}
