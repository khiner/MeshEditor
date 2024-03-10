#version 450

// Based on https://asliceofrendering.com/scene%20helper/2020/01/05/InfiniteGrid/

layout(binding = 0) uniform ViewProjNearFarUBO {
    mat4 View, Projection;
    float Near, Far;
} ViewProjection;

layout(location = 0) in vec3 NearPos;
layout(location = 1) in vec3 FarPos;

layout(location = 0) out vec4 OutColor;

const float ScaleFactor = 0.2;

vec4 Grid(vec3 pos_3d, float scale) {
    const vec2 coord = pos_3d.xz * scale * ScaleFactor;
    const vec2 derivative = fwidth(coord);
    const vec2 grid = abs(fract(coord - 0.5) - 0.5) / derivative;
    vec4 color = vec4(0.329, 0.329, 0.329, 1.0 - min(min(grid.x, grid.y), 1));
    // Highlight the axes.
    const float AxisWidth = 0.4;
    const vec2 clipped_deriv = min(derivative, 1);
    if (pos_3d.x >= -AxisWidth * clipped_deriv.x && pos_3d.x <= AxisWidth * clipped_deriv.x) color = vec4(0, 0, 1, 1);
    if (pos_3d.z >= -AxisWidth * clipped_deriv.y && pos_3d.z <= AxisWidth * clipped_deriv.y) color = vec4(1, 0, 0, 1);
    color.a *= 0.55;
    return color;
}

// Assumes `gl_FragDepth` is set to the depth of the fragment in clip space.
float LinearDepth() {
    const float near = ViewProjection.Near, far = ViewProjection.Far;
    const float clip_space_depth = gl_FragDepth * 2.0 - 1.0; // Normalize to [-1, 1].
    const float linear_depth = (2.0 * near * far) / (near + far - clip_space_depth * (far - near)); // Linear value between `Near` and `Far`.
    return linear_depth / far; // Normalize.
}

// Blend the two grids using their alpha values.
vec4 BlendGrids(vec4 grid1, vec4 grid2) {
    const float alpha = 1.0 - (1.0 - grid1.a) * (1.0 - grid2.a);
    const vec3 color = (grid1.rgb * grid1.a + grid2.rgb * grid2.a * (1.0 - grid1.a)) / alpha;
    return vec4(color, alpha);
}

void main() {
    const float t = -NearPos.y / (FarPos.y - NearPos.y);
    const vec3 pos_3d = NearPos + t * (FarPos - NearPos);
    const vec4 clip_space_pos = ViewProjection.Projection * ViewProjection.View * vec4(pos_3d.xyz, 1);
    gl_FragDepth = clip_space_pos.z / clip_space_pos.w;

    OutColor = BlendGrids(Grid(pos_3d, 10), Grid(pos_3d, 1)) * float(t > 0); // Draw grid at two scales.
    OutColor.a *= (0.5 * (1 - LinearDepth())); // Fade out at the edge of the grid.
}
