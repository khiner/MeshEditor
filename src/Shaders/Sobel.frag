#version 450

layout(binding = 0) uniform sampler2D Tex;
layout(binding = 2) uniform SilhouetteControlsUBO {
    vec4 Color;
    float Thickness;
    float Threshold;
} Controls;

layout(location = 0) in vec2 TexCoord;
layout(location = 0) out vec4 OutColor;

// Sobel kernel for edge detection.
// `Gy` is the transpose of `Gx`.
const mat3 Gx = mat3(
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1
);

void main() {
    const vec4 silhouette_color = texture(Tex, TexCoord);
    const vec2 pixel_size = Controls.Thickness / vec2(textureSize(Tex, 0));
    vec2 sobel = vec2(0);
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            // The alpha channel of the silhouette texture is equal to 1 if the pixel is part of the silhouette, and 0 otherwise.
            const float alpha = texture(Tex, TexCoord + vec2(i, j) * pixel_size).a;
            sobel += vec2(Gx[i + 1][j + 1], Gx[j + 1][i + 1]) * alpha;
        }
    }

    float edge_amount = length(sobel);
    if (edge_amount >= Controls.Threshold) {
        float alpha_weight = min(edge_amount / 2, 1); // This provides a bit of "anti-aliasing" for the edges. (Not a proper normalization factor.)
        OutColor = vec4(Controls.Color.rgb, Controls.Color.a * alpha_weight);
    } else {
        discard;
    }
}
