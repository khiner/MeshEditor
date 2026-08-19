#ifndef VIEWPORTCOMPOSITE_MSL
#define VIEWPORTCOMPOSITE_MSL

#include "Bindless.metal"
#include "Varyings.metal"
#include "tonemapping.metal"

// Packed to match the host struct's layout.
struct ViewportCompositePushConstants {
    uint SceneColorSamplerSlot;
    uint OverlayColorSamplerSlot;
    uint LineDataSamplerSlot;
    // The active view transform: 0 encodes only, 1 tone maps first, 2 passes debug values through.
    uint ViewTransform;
    uint HasOverlay; // Zero when the overlay pass drew nothing, leaving the layer transparent.
    packed_float4 Backdrop; // Display-referred viewport background.
};

inline float2 decode_line_dir(float2 encoded) { return encoded * 2.0f - 1.0f; }

inline float decode_line_dist(float encoded) { return (encoded - 0.6f) * 4.0f; }

// Blender AA constants (overlay_shader_shared.hh).
constant float DISC_RADIUS = 0.5641895835477563f * 1.05f; // M_1_SQRTPI * 1.05
constant float LINE_SMOOTH_START = 0.5f - DISC_RADIUS;
constant float LINE_SMOOTH_END = 0.5f + DISC_RADIUS;

// Center plus the four orthogonal neighbours, which is how far line coverage expands.
constant float2 CompositeOffsets[5] = {
    float2(0, 0), float2(-1, 0), float2(1, 0), float2(0, -1), float2(0, 1)
};

// The overlay layer at `uv`, anti-aliased by borrowing the strongest line covering this pixel.
inline float4 ResolveOverlay(const thread Scene &scene, constant ViewportCompositePushConstants &pc, float2 uv) {
    const float2 texel_size = 1.0f / float2(scene.View.ViewportSize);

    float max_coverage = 0.0f;
    float2 best_uv = uv;

    for (int i = 0; i < 5; i++) {
        const float2 n_uv = uv + CompositeOffsets[i] * texel_size;
        const float4 n_line = scene.SampleTex(pc.LineDataSamplerSlot, n_uv);
        if (n_line.a < 0.5f) continue; // No line data at this sample.

        const float2 perp = decode_line_dir(n_line.xy);
        // The distance from the current pixel to the line that passed through sample i.
        const float curr_dist = decode_line_dist(n_line.z) - dot(perp, CompositeOffsets[i]);
        const float cov = smoothstep(LINE_SMOOTH_END, LINE_SMOOTH_START, abs(curr_dist) - (scene.Theme.EdgeWidth - 0.5f));
        if (cov > max_coverage) {
            max_coverage = cov;
            best_uv = n_uv;
        }
    }

    // The overlay layer is premultiplied, so alpha interpolates along with color: borrowing a
    // neighbour's line at partial coverage yields that line's color at that coverage.
    const float4 overlay = scene.SampleTex(pc.OverlayColorSamplerSlot, uv);
    if (max_coverage == 0.0f) return overlay;
    return mix(overlay, scene.SampleTex(pc.OverlayColorSamplerSlot, best_uv), max_coverage);
}

fragment float4 ViewportCompositeFragment(
    QuadVaryings in [[stage_in]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant ViewportCompositePushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    const float4 overlay = pc.HasOverlay != 0u ? ResolveOverlay(scene, pc, in.TexCoord) : float4(0.0f);

    // The scene layer is premultiplied. Recover radiance for the view transform, then re-apply
    // coverage, so partial-coverage pixels keep their weight.
    const float4 scene_color = scene.SampleTex(pc.SceneColorSamplerSlot, in.TexCoord);
    const float3 radiance = scene_color.a > 0.0f ? scene_color.rgb / scene_color.a : float3(0.0f);
    const float3 scene_display = pc.ViewTransform == 2u ? radiance :
        pc.ViewTransform == 1u ? linearToDisplay(radiance) : linearTosRGB(radiance);
    // The backdrop is UI. It sits under the scene in display space, and the view transform skips it.
    const float3 base = scene_display * scene_color.a + float4(pc.Backdrop).rgb * (1.0f - scene_color.a);
    return float4(base * (1.0f - overlay.a) + overlay.rgb, 1.0f);
}

#endif
