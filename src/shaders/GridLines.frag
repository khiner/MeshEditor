#version 450

// Based on https://asliceofrendering.com/scene%20helper/2020/01/05/InfiniteGrid/

#include "SceneUBO.glsl"

layout(location = 0) in vec3 NearPos;
layout(location = 1) in vec3 FarPos;

layout(location = 0) out vec4 Color;

vec4 Grid(vec3 pos_3d, float scale) {
    const float AxisWidth = 0.5;
    const float LineColor = 0.4;
    const float ScaleFactor = 0.2;
    const vec3 XAxisColor = vec3(1.0, 0.2, 0.332);
    const vec3 ZAxisColor = vec3(0.157, 0.565, 1.0);

    const vec2 coord = pos_3d.xz * scale * ScaleFactor;
    const vec2 d = fwidth(coord);
    const vec2 grid = abs(fract(coord - 0.5) - 0.5) / d;
    const vec2 clipped_deriv = min(d, 1.0);

    const vec2 fades = exp(-pow(abs(pos_3d.xz) / (AxisWidth * clipped_deriv), vec2(5.0)));
    const float xfade = fades.y;
    const float zfade = fades.x;
    const vec3 axis_color = mix(XAxisColor, ZAxisColor, step(xfade, zfade));
    const float axis_alpha = max(xfade, zfade);

    // Fade when lines become subpixel to prevent Moiré.
    const float derivative_fade = 1 - smoothstep(0.2, 1, max(d.x, d.y));

    const float grid_alpha = (1 - min(min(grid.x, grid.y), 1)) * 0.85 * derivative_fade;
    return vec4(mix(vec3(LineColor), axis_color, axis_alpha), max(grid_alpha, axis_alpha));
}

// Assumes `gl_FragDepth` is set to the depth of the fragment in clip space.
float LinearDepth() {
    const float near = SceneViewUBO.CameraNear;
    const float far = SceneViewUBO.CameraFar;
    const float clip_space_depth = gl_FragDepth * 2.0 - 1.0; // Normalize to [-1, 1].
    const float linear_depth = (2.0 * near * far) / (near + far - clip_space_depth * (far - near));
    return linear_depth / far; // Normalize.
}

// Blend the two grids using their alpha values.
vec4 BlendGrids(vec4 a, vec4 b) {
    const float alpha = 1.0 - (1.0 - a.a) * (1.0 - b.a);
    const vec3 c = (a.rgb * a.a + b.rgb * b.a * (1.0 - a.a)) / alpha;
    return vec4(c, alpha);
}

void main() {
    const float t = -NearPos.y / (FarPos.y - NearPos.y);
    const vec3 pos_3d = NearPos + t * (FarPos - NearPos);
    const vec4 clip_space_pos = SceneViewUBO.Proj * SceneViewUBO.View * vec4(pos_3d.xyz, 1);
    gl_FragDepth = clip_space_pos.z / clip_space_pos.w;

    // Draw grid at three scales.
    Color = BlendGrids(BlendGrids(Grid(pos_3d, 10), Grid(pos_3d, 1)), Grid(pos_3d, 0.1)) * float(t > 0);
    Color.a *= (0.6 * (1 - LinearDepth())); // Fade out at the edge of the grid.
}
