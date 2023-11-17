#version 450

layout(binding = 0) uniform sampler2D MainSceneTex;
layout(binding = 1) uniform sampler2D SilhouetteTex;
layout(binding = 3) uniform SilhouetteControlsUBO {
    vec4 Color;
    float Thickness;
    float Threshold;
} Controls;

layout(location = 0) in vec2 TexCoord;
layout(location = 0) out vec4 OutColor;

// Sobel kernel for edge detection.
const mat3 Gx = mat3(
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1
);
// `Gy` is the transpose of `Gx`, but we write it explicitly since globals must be compile-time constants.
const mat3 Gy = mat3(
    -1, -2, -1,
     0,  0,  0,
     1,  2,  1
);

void main() {
    const vec4 main_color = texture(MainSceneTex, TexCoord);
    const vec4 silhouette_color = texture(SilhouetteTex, TexCoord);
    const vec2 pixel_size = Controls.Thickness / vec2(textureSize(SilhouetteTex, 0));
    vec2 sobel = vec2(0);
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            // The alpha channel of the silhouette texture is equal to 1 if the pixel is part of the silhouette, and 0 otherwise.
            const float alpha = texture(SilhouetteTex, TexCoord + vec2(i, j) * pixel_size).a;
            sobel += vec2(Gx[i + 1][j + 1], Gy[i + 1][j + 1]) * alpha;
        }
    }

    OutColor = length(sobel) > Controls.Threshold ? mix(main_color, vec4(Controls.Color.rgb, 1), Controls.Color.a) : main_color;
}
