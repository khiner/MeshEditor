#version 450

layout(binding = 0) uniform sampler2D SilhouetteSampler; // Assumes {Depth, ObjectID} at each pixel.
layout(location = 0) in vec2 TexCoord;
layout(location = 0) out float Out; // ObjectID for edge pixels, discarded otherwise.

const int EdgeHalfWidth = 2;

void main() {
    const ivec2 tex_size = textureSize(SilhouetteSampler, 0);
    const ivec2 texel = ivec2(TexCoord * tex_size);
    const vec2 depth_id = texelFetch(SilhouetteSampler, texel, 0).xy;

    vec2 min_depth_id = vec2(10, 0); // Find the minimum depth (closest) of all neighborhood pixels.
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            if (i == 0 && j == 0) continue;

            // We use vk::SamplerAddressMode::eClampToEdge, so there's no need to check for out-of-bounds.
            const ivec2 neighbor_texel = texel + ivec2(i, j) * EdgeHalfWidth;
            const vec2 neighbor_depth_id = texelFetch(SilhouetteSampler, neighbor_texel, 0).xy;
            if (depth_id.y != neighbor_depth_id.y) {
                const vec2 edge_depth_id = depth_id.x != 0 ? depth_id : neighbor_depth_id;
                min_depth_id = vec2(min(min_depth_id.x, edge_depth_id.x), edge_depth_id.y);
            }
        }
    }

    if (min_depth_id.y != 0) {
        gl_FragDepth = min_depth_id.x;
        Out = min_depth_id.y;
    } else {
        discard;
    }
}
