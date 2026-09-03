#ifndef GRIDLINES_MSL
#define GRIDLINES_MSL

#include "Bindless.metal"
#include "Varyings.metal"

// Infinite vertices cover the y = 0 plane while preserving rasterized depth for early depth testing.
constant float R3 = 0.86602540378f; // sin(120 degrees)
constant float4 GridVerts[9] = {
    float4(0, 0, 0, 1), float4(1, 0, 0, 0), float4(-0.5f, 0, R3, 0),
    float4(0, 0, 0, 1), float4(-0.5f, 0, R3, 0), float4(-0.5f, 0, -R3, 0),
    float4(0, 0, 0, 1), float4(-0.5f, 0, -R3, 0), float4(1, 0, 0, 0)
};

vertex GridVaryings GridLinesVertex(
    uint vertex_id [[vertex_id]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    const float4 plane_pos = GridVerts[vertex_id];
    return GridVaryings{scene.ViewProj() * plane_pos, plane_pos};
}

// Returns anti-aliased line intensity for spacing 1 / scale.
inline float GridLineIntensity(float3 pos, float scale, float2 camera_xz, float2 world_d) {
    // Offset by the nearest grid line to preserve float precision at large distances.
    const float2 coord = pos.xz * scale - round(camera_xz * scale);
    const float2 d = world_d * scale;
    const float2 grid = abs(fract(coord - 0.5f) - 0.5f) / d;
    // Fade subpixel lines to prevent moire.
    const float moire_fade = 1.0f - smoothstep(0.2f, 1.0f, max(d.x, d.y));
    return (1.0f - min(min(grid.x, grid.y), 1.0f)) * moire_fade;
}

// Composites premultiplied `a` over `b`.
inline float4 BlendGrids(float4 a, float4 b) {
    const float alpha = 1.0f - (1.0f - a.a) * (1.0f - b.a);
    if (alpha < 0.001f) return float4(0);
    return float4((a.rgb * a.a + b.rgb * b.a * (1.0f - a.a)) / alpha, alpha);
}

fragment OverlayTargets GridLinesFragment(
    GridVaryings in [[stage_in]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    constant ViewportThemeColors &colors = scene.Theme.Colors;
    const float3 pos_3d = in.PlanePos.xyz / in.PlanePos.w;
    const float3 camera_position = float3(view.CameraPosition);

    const float3 to_camera = camera_position - pos_3d;
    const float dist = length(to_camera);
    const float3 V = to_camera / dist;

    // Select three grid scales from the camera height.
    const float camera_height = abs(camera_position.y);
    const float log_dist = log(max(camera_height, 0.001f)) / log(10.0f);
    const float base_level = floor(log_dist);

    const float base_scale = pow(10.0f, -base_level);
    const float fine_scale = base_scale * 10.0f;
    const float emph_scale = base_scale * 0.1f;

    const float2 camera_xz = camera_position.xz;
    const float2 world_d = fwidth(pos_3d.xz);
    const float i_fine = GridLineIntensity(pos_3d, fine_scale, camera_xz, world_d);
    const float i_base = GridLineIntensity(pos_3d, base_scale, camera_xz, world_d);
    const float i_emph = GridLineIntensity(pos_3d, emph_scale, camera_xz, world_d);

    const float frac = fract(log_dist);
    const float4 grid_color = float4(colors.GridLine);
    const float4 emph_color = float4(colors.GridEmphasis);
    const float4 fine_grid = float4(grid_color.rgb, i_fine * grid_color.a * (1.0f - frac));
    const float4 base_grid = float4(mix(grid_color.rgb, emph_color.rgb, 1.0f - frac), i_base * mix(grid_color.a, emph_color.a, 1.0f - frac));
    const float4 emph_grid = float4(emph_color.rgb, i_emph * emph_color.a);
    const float4 grid = BlendGrids(emph_grid, BlendGrids(base_grid, fine_grid));

    const float x_axis = 1.0f - smoothstep(0.0f, 1.5f, abs(pos_3d.z) / world_d.y);
    const float z_axis = 1.0f - smoothstep(0.0f, 1.5f, abs(pos_3d.x) / world_d.x);
    const float3 axis_color = mix(float3(colors.GridAxisX), float3(colors.GridAxisZ), step(x_axis, z_axis));

    float4 color = BlendGrids(float4(axis_color, max(x_axis, z_axis)), grid);
    // Fade the grid near horizontal viewing angles.
    color.a *= 1.0f - pow(1.0f - abs(V.y), 3.0f);
    // Fade toward the far clip plane.
    color.a *= 1.0f - smoothstep(0.0f, 0.5f * view.CameraFar, dist - 0.5f * view.CameraFar);
    return OverlayTargets{color, float4(0)};
}

#endif
