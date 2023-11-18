#version 450

layout(binding = 0) uniform sampler2D Tex;
layout(location = 0) in vec2 TexCoord;
layout(location = 0) out vec4 OutColor;

// Sobel kernel for edge detection.
// `Gy` is the transpose of `Gx`.
const mat3 Gx = mat3(
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1
);

const float Thickness = 2;
const float Threshold = 0;

void main() {
    const vec4 silhouette_color = texture(Tex, TexCoord);
    const vec2 pixel_size = Thickness / vec2(textureSize(Tex, 0));
    vec2 sobel = vec2(0);
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            // The alpha channel of the silhouette texture is equal to 1 if the pixel is part of the silhouette, and 0 otherwise.
            const vec4 neighbor_color = texture(Tex, TexCoord + vec2(i, j) * pixel_size);
            sobel += vec2(Gx[i + 1][j + 1], Gx[j + 1][i + 1]) * neighbor_color.a;
        }
    }

    float edge_amount = length(sobel);
    if (edge_amount > Threshold) {
        float min_depth = 1;
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                const vec4 neighbor_color = texture(Tex, TexCoord + vec2(i, j) * pixel_size);
                if (neighbor_color.a == 1) {
                    min_depth = min(min_depth, neighbor_color.r);
                }
            }
        }
        float alpha = min(edge_amount / 2, 1); // This provides a bit of "anti-aliasing" for the edges. (Not a proper normalization factor.)
        OutColor = vec4(min_depth, 1, 1, alpha);
    } else {
        discard;
    }
}
