#version 450

layout(binding = 0) uniform sampler2D Tex; // Assumes each pixel of the texture contains {depth, *, *, HasGeometry}
layout(location = 0) in vec2 TexCoord;
layout(location = 0) out vec4 OutColor; // {IsEdge, Depth, Depth, (Normalized) amount}

const float EdgeThicknessPixels = 4;

void main() {
    const vec4 this_value = texture(Tex, TexCoord);
    const bool has_geometry = this_value.a == 1;
    const vec2 neighborhood_half_extent = (EdgeThicknessPixels / 2.0) / vec2(textureSize(Tex, 0));

    float min_depth = 100; // Find the minimum depth (closest) of all neighborhood pixels within the geometry.
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            if (i == 0 && j == 0) continue;
            const vec4 neighbor_value = texture(Tex, TexCoord + vec2(i, j) * neighborhood_half_extent);
            const bool neighbor_has_geometry = neighbor_value.a == 1;
            if ((!has_geometry && neighbor_has_geometry) || (has_geometry && !neighbor_has_geometry)) {
                const float geometry_depth = has_geometry ? this_value.r : neighbor_value.r;
                min_depth = min(min_depth, geometry_depth);
            }
        }
    }

    if (min_depth < 90) {
        OutColor = vec4(1, min_depth, min_depth, 1); // Output format: {IsEdge, Depth, Depth, EdgeWeight}
    } else {
        discard;
    }
}
