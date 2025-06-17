#version 450

// Based on https://asliceofrendering.com/scene%20helper/2020/01/05/InfiniteGrid/

layout(binding = 0) uniform ViewProjNearFarUBO {
    mat4 View, Projection;
    float Near, Far;
} ViewProjection;

layout(location = 0) in vec3 NearPos;
layout(location = 1) in vec3 FarPos;

layout(location = 0) out vec4 Color;

const float ScaleFactor = 0.2;
const float Gray = 0.45; // Used for the grid lines.

vec4 Grid(vec3 pos_3d, float scale) {
    const vec2 coord = pos_3d.xz * scale * ScaleFactor;
    const vec2 derivative = fwidth(coord);
    const vec2 grid = abs(fract(coord - 0.5) - 0.5) / derivative;
    // Highlight the axes.
    const float AxisWidth = 0.4;
    const vec2 clipped_deriv = min(derivative, 1);
    return pos_3d.x >= -AxisWidth * clipped_deriv.x && pos_3d.x <= AxisWidth * clipped_deriv.x ? vec4(0.17, 0.56, 1, 1) :
           pos_3d.z >= -AxisWidth * clipped_deriv.y && pos_3d.z <= AxisWidth * clipped_deriv.y ? vec4(1, 0.21, 0.32, 1) :
           vec4(Gray, Gray, Gray, (1.0 - min(min(grid.x, grid.y), 1)) * 0.6);
}

// Assumes `gl_FragDepth` is set to the depth of the fragment in clip space.
float LinearDepth() {
    const float near = ViewProjection.Near, far = ViewProjection.Far;
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
    const vec4 clip_space_pos = ViewProjection.Projection * ViewProjection.View * vec4(pos_3d.xyz, 1);
    gl_FragDepth = clip_space_pos.z / clip_space_pos.w;

    // Draw grid at three scales.
    Color = BlendGrids(BlendGrids(Grid(pos_3d, 10), Grid(pos_3d, 1)), Grid(pos_3d, 0.1)) * float(t > 0);
    Color.a *= (0.6 * (1 - LinearDepth())); // Fade out at the edge of the grid.
}
