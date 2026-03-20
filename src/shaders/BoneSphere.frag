#version 450

#include "SceneUBO.glsl"
#include "ViewportTheme.glsl"

layout(location = 0) flat in vec3 SphereCenter; // View-space
layout(location = 2) in vec3 ViewPos; // View-space fragment position on billboard
layout(location = 3) flat in vec4 BoneColor;
layout(location = 4) flat in vec4 StateColor;
layout(location = 5) flat in float SphereRadius;

layout(location = 0) out vec4 OutColor;
layout(location = 1) out vec4 OutLineData;

void main() {
    // Ray-sphere intersection in view space
    const vec3 ray_dir = normalize(ViewPos);
    const vec3 oc = ViewPos - SphereCenter;

    const float b = dot(oc, ray_dir);
    const float c = dot(oc, oc) - SphereRadius * SphereRadius;
    const float discriminant = b * b - c;
    if (discriminant < 0.0) discard;

    const float t = -sqrt(discriminant) - b;
    const vec3 hit_view = ViewPos + ray_dir * t;
    const vec3 normal = normalize(hit_view - SphereCenter);

    // Blender-style angled lighting (same as BoneSolid)
    const vec3 light = normalize(vec3(0.1, 0.1, 0.8));
    const float fac = clamp(dot(normal, light) * 0.8 + 0.2, 0.0, 1.0);
    const vec3 color = mix(StateColor.rgb, BoneColor.rgb, fac * fac);

    const float alpha = (SceneViewUBO.BoneXRay != 0u) ? 0.4 : 1.0;
    OutColor = vec4(color, alpha);
    OutLineData = vec4(0); // Not a line

    // Correct depth: transform hit back to world space, then project
    const vec3 world_hit = transpose(SceneViewUBO.ViewRotation) * hit_view + SceneViewUBO.CameraPosition;
    const vec4 clip = SceneViewUBO.ViewProj * vec4(world_hit, 1.0);
    // X-ray: write near-far-plane depth so fills don't occlude wires (but still pass the eLess depth test against cleared 1.0).
    gl_FragDepth = (SceneViewUBO.BoneXRay != 0u) ? 0.999999 : clip.z / clip.w;
}
