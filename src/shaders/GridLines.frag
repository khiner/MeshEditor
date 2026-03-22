#version 450

// Infinite grid matching Blender's grid appearance

#include "SceneUBO.glsl"
#include "ViewportTheme.glsl"

layout(location = 0) in vec3 RayOrigin;
layout(location = 1) in vec3 RayDir;

layout(location = 0) out vec4 Color;

// Anti-aliased grid line intensity [0,1] at the given scale.
// Scale determines line spacing: scale=N produces lines every 1/N world units.
// camera_xz offsets the computation for float32 precision at far distances (same idea as Blender's grid_buf.offset).
float GridLineIntensity(vec3 pos, float scale, vec2 camera_xz, vec2 world_d) {
    // Snap camera to nearest grid line so fract() stays precise at far distances.
    const vec2 coord = pos.xz * scale - round(camera_xz * scale);
    const vec2 d = world_d * scale;
    const vec2 grid = abs(fract(coord - 0.5) - 0.5) / d;
    // Fade when lines become subpixel to prevent Moire.
    const float moire_fade = 1.0 - smoothstep(0.2, 1.0, max(d.x, d.y));
    return (1.0 - min(min(grid.x, grid.y), 1.0)) * moire_fade;
}

// Over-compositing: a over b
vec4 BlendGrids(vec4 a, vec4 b) {
    const float alpha = 1.0 - (1.0 - a.a) * (1.0 - b.a);
    if (alpha < 0.001) return vec4(0);
    return vec4((a.rgb * a.a + b.rgb * b.a * (1.0 - a.a)) / alpha, alpha);
}

void main() {
    const float t = -RayOrigin.y / RayDir.y;
    if (t <= 0) discard;

    const vec3 pos_3d = RayOrigin + t * RayDir;
    const vec4 clip_space_pos = SceneViewUBO.ViewProj * vec4(pos_3d, 1);
    gl_FragDepth = clip_space_pos.z / clip_space_pos.w;

    const vec3 to_camera = SceneViewUBO.CameraPosition - pos_3d;
    const float dist = length(to_camera);
    const vec3 V = to_camera / dist;

    // Three dynamic grid scales with levels based on camera height above the grid plane
    const float camera_height = abs(SceneViewUBO.CameraPosition.y);
    const float log_dist = log(max(camera_height, 0.001)) / log(10.0);
    const float base_level = floor(log_dist);

    const float base_scale = pow(10.0, -base_level);
    const float fine_scale = base_scale * 10.0;
    const float emph_scale = base_scale * 0.1;

    const vec2 camera_xz = SceneViewUBO.CameraPosition.xz;
    const vec2 world_d = fwidth(pos_3d.xz);
    const float i_fine = GridLineIntensity(pos_3d, fine_scale, camera_xz, world_d);
    const float i_base = GridLineIntensity(pos_3d, base_scale, camera_xz, world_d);
    const float i_emph = GridLineIntensity(pos_3d, emph_scale, camera_xz, world_d);

    // Per-level emphasis and alpha
    const float frac = fract(log_dist);
    const vec4 grid_color = ViewportTheme.Colors.GridLine;
    const vec4 emph_color = ViewportTheme.Colors.GridEmphasis;
    // Fine: pure grid color, fading out
    const vec4 fine_grid = vec4(grid_color.rgb, i_fine * grid_color.a * (1.0 - frac));
    // Base: color and alpha lerp from grid toward emphasis
    const vec4 base_grid = vec4(mix(grid_color.rgb, emph_color.rgb, 1.0 - frac), i_base * mix(grid_color.a, emph_color.a, 1.0 - frac));
    // Emphasis: pure emphasis color, full alpha
    const vec4 emph_grid = vec4(emph_color.rgb, i_emph * emph_color.a);
    // Composite: emphasis on top (highest priority), then base, then fine
    const vec4 grid = BlendGrids(emph_grid, BlendGrids(base_grid, fine_grid));

    // Axis lines
    const float x_axis = 1.0 - smoothstep(0.0, 1.5, abs(pos_3d.z) / max(world_d.y, 0.001));
    const float z_axis = 1.0 - smoothstep(0.0, 1.5, abs(pos_3d.x) / max(world_d.x, 0.001));
    const vec3 axis_color = mix(ViewportTheme.Colors.GridAxisX, ViewportTheme.Colors.GridAxisZ, step(x_axis, z_axis));

    // Axes on top of grid
    Color = BlendGrids(vec4(axis_color, max(x_axis, z_axis)), grid);
    // Steep angle fade: grid disappears when viewed nearly horizontally
    Color.a *= 1.0 - pow(1.0 - abs(V.y), 3);
    // Camera clip fade: grid fades toward far clip plane
    Color.a *= 1.0 - smoothstep(0.0, 0.5 * SceneViewUBO.CameraFar, dist - 0.5 * SceneViewUBO.CameraFar);
}
