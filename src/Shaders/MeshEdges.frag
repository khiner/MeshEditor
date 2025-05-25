#version 450

layout(binding = 0) uniform sampler2D Tex; // Assumes each pixel of the texture contains {depth, ObjectID}
layout(location = 0) in vec2 TexCoord;
layout(location = 0) out vec4 OutColor; // {IsEdge, Depth, Depth, (Normalized) amount}

const float EdgeThicknessPixels = 4;

void main() {
    const vec2 this_value = texture(Tex, TexCoord).rg;
    const float this_id = this_value.g;
    const vec2 neighborhood_half_extent = (EdgeThicknessPixels / 2.0) / vec2(textureSize(Tex, 0));

    float min_depth = 100; // Find the minimum depth (closest) of all neighborhood pixels within the mesh.
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            if (i == 0 && j == 0) continue;
            const vec2 neighbor_value = texture(Tex, TexCoord + vec2(i, j) * neighborhood_half_extent).rg;
            const float neighbor_id = neighbor_value.g;
            if (this_id != neighbor_id) {
                const float mesh_depth = this_value.r > 0 ? this_value.r : neighbor_value.r;
                min_depth = min(min_depth, mesh_depth);
            }
        }
    }

    if (min_depth < 100) {
        OutColor = vec4(1, min_depth, min_depth, 1);
    } else {
        discard;
    }
}
