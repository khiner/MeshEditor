#version 450

#include "SceneUBO.glsl"
#include "WorkbenchLightingUBO.glsl"
#include "tonemapping.glsl"

layout(location = 0) in vec3 WorldNormal;
layout(location = 1) in vec3 WorldPosition;
layout(location = 2) in vec4 InColor;
layout(location = 3) flat in uint FaceOverlayFlags;
layout(location = 11) in vec3 ViewNormal;

layout(location = 0) out vec4 OutColor;

// Wrapped diffuse: softens the light/dark terminator (Blender: wrapped_lighting).
float wrapped_lighting(float NL, float w) {
    float w1 = w + 1.0;
    return clamp((NL + w) / (w1 * w1), 0.0, 1.0);
}

// Schlick-Fresnel specular approximation (Blender: brdf_approx).
vec3 brdf_approx(vec3 spec_color, float roughness, float NV) {
    float fresnel = exp2(-8.35 * NV) * (1.0 - roughness);
    return mix(spec_color, vec3(1.0), fresnel);
}

// Blinn-Phong specular with energy normalization (Blender: blinn_specular).
float blinn_specular(float shininess, float spec_angle, float NL) {
    return pow(spec_angle, shininess) * NL * (shininess * 0.125 + 1.0);
}

vec3 safe_normalize(vec3 v) {
    const float len2 = dot(v, v);
    return len2 > 1e-12 ? v * inversesqrt(len2) : vec3(0.0);
}

void main() {
    const vec3 I_ws = normalize(SceneViewUBO.CameraPosition - WorldPosition);
    const vec3 I_vs = SceneViewUBO.ViewRotation * I_ws;
    const vec3 N0 = safe_normalize(ViewNormal);
    const vec3 N = faceforward(N0, -I_vs, N0);
    // Perfect mirror reflection of view direction about N, for large-area specular simulation.
    const vec3 R = -reflect(I_vs, N);

    const float roughness = 0.5;
    const float metallic = 0.0;
    const float NV = max(dot(N, I_vs), 0.0);

    const vec3 base_color = InColor.rgb;
    const vec3 spec_color = mix(vec3(0.05), base_color, metallic); // vec3(0.05) for non-metallic

    // Both accumulate starting from ambient (Blender: get_world_lighting).
    vec3 diffuse = WorkbenchLightingUBO.AmbientColor;
    vec3 specular = WorkbenchLightingUBO.AmbientColor;

    for (int i = 0; i < 4; i++) {
        const vec3 L = safe_normalize(WorkbenchLightingUBO.StudioLights[i].Direction);
        const float wrap = WorkbenchLightingUBO.StudioLights[i].Wrap;

        // Per-light shininess: softer lights (higher wrap) get lower gloss.
        // Blender: gloss = (1 - roughness) * (1 - wrap), shininess = exp2(10 * gloss + 1)
        const float gloss = (1.0 - roughness) * (1.0 - wrap);
        const float shininess = exp2(10.0 * gloss + 1.0);

        const float NL = dot(N, L);            // unclamped — wrapped_lighting handles negative values
        const float NL_c = max(NL, 0.0);       // clamped — Blinn-Phong needs non-negative
        const vec3 half_dir = safe_normalize(L + I_vs);
        const float spec_angle = clamp(dot(half_dir, N), 0.0, 1.0);

        // Blend sharp Blinn-Phong with soft environment-like wrap on reflection vector.
        // High wrap -> more environment-like (soft area light). Low wrap -> sharper Blinn.
        const float blinn = blinn_specular(shininess, spec_angle, NL_c);
        const float wrapped_NL = dot(L, R);
        const float spec_env = wrapped_lighting(wrapped_NL, mix(wrap, 1.0, roughness));
        const float spec_per_light = mix(blinn, spec_env, wrap * wrap);

        diffuse  += wrapped_lighting(NL, wrap) * WorkbenchLightingUBO.StudioLights[i].DiffuseColor;
        specular += spec_per_light * WorkbenchLightingUBO.StudioLights[i].SpecularColor;
    }

    // Energy conservation: diffuse attenuated by specular reflectance (Blender: spec_energy).
    const float spec_energy = dot(spec_color, vec3(0.33333));
    diffuse *= base_color * (1.0 - spec_energy);

    vec3 color = diffuse + brdf_approx(spec_color, roughness, NV) * specular;
    if (FaceOverlayFlags != 0u) {
        const bool is_edit_face = SceneViewUBO.InteractionMode == InteractionModeEdit &&
            SceneViewUBO.EditElement == EditElementFace;
        const vec4 selected = is_edit_face ? ViewportTheme.Colors.FaceSelected : ViewportTheme.Colors.FaceSelectedIncidental;
        const vec3 overlay = (FaceOverlayFlags & 2u) != 0u ?
            mix(selected.rgb, ViewportTheme.Colors.ElementActive.rgb, 0.5) :
            selected.rgb;
        color = mix(color, overlay, selected.a);
    }
    OutColor = vec4(linearTosRGB(color), InColor.a);
}
