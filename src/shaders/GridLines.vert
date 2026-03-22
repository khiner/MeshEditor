#version 450

#include "SceneUBO.glsl"

layout(location = 0) out vec3 RayOrigin;
layout(location = 1) out vec3 RayDir;

// Triangle strip.
const vec2 QuadPositions[4] = vec2[](vec2(-1, -1), vec2(1, -1), vec2(-1, 1), vec2(1, 1));

void main() {
    const vec2 p = QuadPositions[gl_VertexIndex];
    // Extract projection diagonal from ViewProj = Proj * View.
    // For both perspective and ortho, mat3(ViewProj) = diag(...) * ViewRotation,
    // so right-multiplying by transpose(ViewRotation) recovers the diagonal.
    const mat3 proj3 = mat3(SceneViewUBO.ViewProj) * transpose(SceneViewUBO.ViewRotation);
    const mat3 inv_rot = transpose(SceneViewUBO.ViewRotation);
    if (SceneViewUBO.ScreenPixelScale < 0.0) {
        // Orthographic: parallel rays, varying origin
        // proj3 diagonal = (1/mag.x, 1/mag.y, ...) for orthoRH_ZO
        RayOrigin = SceneViewUBO.CameraPosition + inv_rot * vec3(p.x / proj3[0][0], p.y / proj3[1][1], 0.0);
        RayDir = inv_rot * vec3(0.0, 0.0, -1.0);
    } else {
        // Perspective: common origin, varying direction
        RayOrigin = SceneViewUBO.CameraPosition;
        RayDir = inv_rot * vec3(p.x / proj3[0][0], p.y / proj3[1][1], -1.0);
    }

    gl_Position = vec4(p, 0, 1);
}
