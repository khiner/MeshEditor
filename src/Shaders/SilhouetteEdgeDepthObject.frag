#version 450

layout(binding = 0) uniform sampler2D SilhouetteSampler; // Assumes {Depth, ObjectID} at each pixel.
layout(location = 0) in vec2 TexCoord;
layout(location = 0) out float Out; // ObjectID for edge pixels, discarded otherwise.

const float EdgeThicknessPixels = 4;

void main() {
    const vec2 this_value = texture(SilhouetteSampler, TexCoord).xy; // {Depth, ObjectID}
    const float this_id = this_value.y;
    const vec2 neighborhood_half_extent = (EdgeThicknessPixels / 2.0) / vec2(textureSize(SilhouetteSampler, 0));

    float min_depth = 100; // Find the minimum depth (closest) of all neighborhood pixels within the mesh.
    uint min_depth_id = 0; // Track the ID of the closest mesh pixel.
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            if (i == 0 && j == 0) continue;

            const vec2 neighbor_value = texture(SilhouetteSampler, TexCoord + vec2(i, j) * neighborhood_half_extent).xy;
            const float neighbor_id = neighbor_value.y;
            if (this_id != neighbor_id) {
                const vec2 value = this_value.x > 0 ? this_value : neighbor_value;
                min_depth = min(min_depth, value.x);
                min_depth_id = uint(value.y);
            }
        }
    }

    if (min_depth < 100) {
        gl_FragDepth = min_depth;
        Out = min_depth_id;
    } else {
        discard;
    }
}
