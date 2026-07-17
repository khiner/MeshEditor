#version 450

#include "SceneUBO.glsl"
#include "Velocity.glsl"

layout(location = 0) in vec2 InNDC;
// Location 1: the scene pass's velocity attachment (the scene color at 0 is masked off).
layout(location = 1) out vec4 OutMotion;

// The background sits at infinity, so only view rotation moves it. Projecting the view ray as a
// direction lands on the point at infinity along it, which is what the background shows.
vec2 ProjectDirToUv(mat4 view_proj, vec3 dir) {
    const vec4 clip = view_proj * vec4(dir, 0.0);
    return clip.xy / clip.w;
}

void main() {
    const mat3 proj3 = mat3(SceneViewUBO.ViewProj) * transpose(SceneViewUBO.ViewRotation);
    const mat3 inv_rot = transpose(SceneViewUBO.ViewRotation);
    const vec3 world_dir = normalize(inv_rot * vec3(InNDC.x / proj3[0][0], InNDC.y / proj3[1][1], -1.0));

    const vec2 curr_uv = ProjectDirToUv(SceneViewUBO.ViewProj, world_dir);
    const vec2 prev_uv = ProjectDirToUv(SceneViewUBO.PrevViewProj, world_dir);
    const vec2 next_uv = ProjectDirToUv(SceneViewUBO.NextViewProj, world_dir);
    OutMotion = PackVelocity(vec4(prev_uv - curr_uv, curr_uv - next_uv) * 0.5);
}
