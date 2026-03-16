#version 450

#include "SceneUBO.glsl"
#include "ViewportTheme.glsl"

layout(location = 0) flat in vec3 SphereCenter; // View-space
layout(location = 1) flat in float SphereRadius;
layout(location = 2) in vec3 ViewPos;            // View-space fragment position on billboard
layout(location = 3) flat in vec4 BoneColor;
layout(location = 4) flat in vec4 StateColor;

layout(location = 0) out vec4 OutColor;
layout(location = 1) out vec4 OutLineData;

void main() {
    // Ray-sphere intersection in view space
    vec3 ray_dir = normalize(ViewPos);
    vec3 oc = ViewPos - SphereCenter;

    float b = dot(oc, ray_dir);
    float c = dot(oc, oc) - SphereRadius * SphereRadius;
    float discriminant = b * b - c;

    if (discriminant < 0.0) discard;

    float t = -sqrt(discriminant) - b;
    vec3 hit_view = ViewPos + ray_dir * t;
    vec3 normal = normalize(hit_view - SphereCenter);

    // Blender-style angled lighting (same as BoneSolid)
    vec3 light = normalize(vec3(0.1, 0.1, 0.8));
    float fac = clamp(dot(normal, light) * 0.8 + 0.2, 0.0, 1.0);
    vec3 color = mix(StateColor.rgb, BoneColor.rgb, fac * fac);

    OutColor = vec4(color, 1.0);
    OutLineData = vec4(0); // Not a line

    // Correct depth: transform hit back to world space, then project
    // View = [ViewRotation | -ViewRotation * CameraPosition]
    // inverse(View) * hit_view = transpose(ViewRotation) * hit_view + CameraPosition
    vec3 world_hit = transpose(SceneViewUBO.ViewRotation) * hit_view + SceneViewUBO.CameraPosition;
    vec4 clip = SceneViewUBO.ViewProj * vec4(world_hit, 1.0);
    gl_FragDepth = clip.z / clip.w;
}
