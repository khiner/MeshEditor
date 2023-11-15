#version 450

// Based on https://asliceofrendering.com/scene%20helper/2020/01/05/InfiniteGrid/

layout(location = 0) in vec3 NearPos;
layout(location = 1) in vec3 FarPos;

layout(location = 0) out vec4 OutColor;

layout(binding = 1) uniform ViewProjNearFarUBO {
    mat4 View;
    mat4 Projection;
    float Near;
    float Far;
} ViewProjNearFar;

vec4 Grid(vec3 pos_3d, float scale) {
    vec2 coord = pos_3d.xz * scale;
    vec2 derivative = fwidth(coord);
    vec2 grid = abs(fract(coord - 0.5) - 0.5) / derivative;
    vec2 clipped_deriv = min(derivative, 1);
    vec4 color = vec4(0.2, 0.2, 0.2, 1.0 - min(min(grid.x, grid.y), 1));
    // Highlight the axes.
    if (pos_3d.x > -0.1 * clipped_deriv.x && pos_3d.x < 0.1 * clipped_deriv.x) color.b = 1;
    if (pos_3d.z > -0.1 * clipped_deriv.y && pos_3d.z < 0.1 * clipped_deriv.y) color.r = 1;
    return color;
}

// Assumes `gl_FragDepth` is set to the depth of the fragment in clip space.
float LinearDepth() {
    float near = ViewProjNearFar.Near;
    float far = ViewProjNearFar.Far;
    float clip_space_depth = gl_FragDepth * 2.0 - 1.0; // Normalize to [-1, 1].
    float linear_depth = (2.0 * near * far) / (near + far - clip_space_depth * (far - near)); // Linear value between `Near` and `Far`.
    return linear_depth / far; // Normalize.
}

void main() {
    float t = -NearPos.y / (FarPos.y - NearPos.y);
    vec3 pos_3d = NearPos + t * (FarPos - NearPos);
    vec4 clip_space_pos = ViewProjNearFar.Projection * ViewProjNearFar.View * vec4(pos_3d.xyz, 1);
    gl_FragDepth = clip_space_pos.z / clip_space_pos.w;

    OutColor = (Grid(pos_3d, 4) + Grid(pos_3d, 1)) * float(t > 0); // Draw grid at two scales.
    OutColor.a *= 0.5 * (1 - smoothstep(0.8, 1, LinearDepth())); // Fade out at the edge of the grid.
}
