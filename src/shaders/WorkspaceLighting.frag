#version 450

// Blender-matching solid viewport lighting. Ported from workbench_world_light.bsl.hh.

#include "SceneUBO.glsl"
#include "WorkspaceLights.glsl"
#include "tonemapping.glsl"

layout(location = 0) in vec3 WorldNormal;
layout(location = 1) in vec3 WorldPosition;
layout(location = 2) in vec4 InColor;
layout(location = 3) flat in uint FaceOverlayFlags;

layout(location = 0) out vec4 OutColor;

// Approximate Fresnel effect with roughness attenuation
vec3 brdf_approx(vec3 spec_color, float roughness, float NV) {
    float fresnel = exp2(-8.35 * NV) * (1.0 - roughness);
    return mix(spec_color, vec3(1.0), fresnel);
}

// Normalized Blinn-Phong specular
vec4 blinn_specular(vec4 shininess, vec4 spec_angle, vec4 NL) {
    vec4 normalization_factor = shininess * 0.125 + 1.0; // = (shininess + 8.0) / (8.0 * PI)
    return pow(spec_angle, shininess) * NL * normalization_factor;
}

// Wrapped lighting: NL unclamped, w in [0..1]
vec4 wrapped_lighting(vec4 NL, vec4 w) {
    vec4 w_1 = w + 1.0;
    return clamp((NL + w) / (w_1 * w_1), 0.0, 1.0);
}

vec3 get_world_lighting(vec3 base_color, float roughness, float metallic, vec3 N, vec3 I) {
    vec3 specular_color, diffuse_color;
    if (WorkspaceLights.UseSpecular != 0u) {
        diffuse_color = mix(base_color, vec3(0.0), metallic);
        specular_color = mix(vec3(0.05), base_color, metallic);
    } else {
        diffuse_color = base_color;
        specular_color = vec3(0.0);
    }

    vec3 specular_light = WorkspaceLights.AmbientColor;
    vec3 diffuse_light = WorkspaceLights.AmbientColor;
    vec4 wrap = vec4(
        WorkspaceLights.Lights[0].Wrap,
        WorkspaceLights.Lights[1].Wrap,
        WorkspaceLights.Lights[2].Wrap,
        WorkspaceLights.Lights[3].Wrap
    );

    if (WorkspaceLights.UseSpecular != 0u) {
        vec3 R = -reflect(I, N);
        vec4 spec_angle, spec_NL, wrapped_NL;
        for (int i = 0; i < 4; i++) {
            vec3 L = WorkspaceLights.Lights[i].Direction;
            vec3 half_dir = normalize(L + I);
            wrapped_NL[i] = dot(L, R);
            spec_angle[i] = clamp(dot(half_dir, N), 0.0, 1.0);
            spec_NL[i] = clamp(dot(L, N), 0.0, 1.0);
        }

        vec4 gloss = vec4(1.0 - roughness);
        // Reduce gloss for smooth light (simulate bigger light).
        gloss *= 1.0 - wrap;
        vec4 shininess = exp2(10.0 * gloss + 1.0);
        vec4 spec_light = blinn_specular(shininess, spec_angle, spec_NL);

        // Simulate environment light.
        vec4 spec_env = wrapped_lighting(wrapped_NL, mix(wrap, vec4(1.0), roughness));
        spec_light = mix(spec_light, spec_env, wrap * wrap);

        // Multiply by per-light specular colors.
        for (int i = 0; i < 4; i++) {
            specular_light += spec_light[i] * WorkspaceLights.Lights[i].SpecularColor;
        }

        specular_color = brdf_approx(specular_color, roughness, clamp(dot(N, I), 0.0, 1.0));
    }
    specular_light *= specular_color;

    // Diffuse: evaluate 4 lights with wrapped lighting.
    vec4 diff_NL;
    for (int i = 0; i < 4; i++) {
        diff_NL[i] = dot(WorkspaceLights.Lights[i].Direction, N);
    }

    vec4 diff_light = wrapped_lighting(diff_NL, wrap);
    for (int i = 0; i < 4; i++) {
        diffuse_light += diff_light[i] * WorkspaceLights.Lights[i].DiffuseColor;
    }

    // Energy conservation: reduce diffuse by specular energy.
    float spec_energy = dot(specular_color, vec3(0.33333));
    diffuse_light *= diffuse_color * (1.0 - spec_energy);

    return diffuse_light + specular_light;
}

void main() {
    // Transform normal and view direction to view space (camera-relative lighting, matching Blender default).
    vec3 N = normalize(SceneViewUBO.ViewRotation * WorldNormal);
    vec3 view_dir = SceneViewUBO.CameraPosition - WorldPosition;
    N = faceforward(N, -(SceneViewUBO.ViewRotation * view_dir), N);
    vec3 I = normalize(SceneViewUBO.ViewRotation * view_dir);

    // Blender's workbench defaults: metallic=0.0, roughness=0.4.
    // pack_data() applies Disney roughness remap (sqrt) before the shader sees it.
    vec3 color = get_world_lighting(InColor.rgb, sqrt(0.4), 0.0, N, I);
    if (FaceOverlayFlags != 0u) {
        // Theme colors in ViewportTheme are sRGB - convert to linear for blending.
        const bool is_edit_face = SceneViewUBO.InteractionMode == InteractionMode_Edit &&
            SceneViewUBO.EditElement == Element_Face;
        const vec4 selected = is_edit_face ? ViewportTheme.Colors.FaceSelected : ViewportTheme.Colors.FaceSelectedIncidental;
        const vec3 overlay = (FaceOverlayFlags & 2u) != 0u ?
            mix(sRGBToLinear(selected.rgb), sRGBToLinear(ViewportTheme.Colors.ElementActive.rgb), 0.5) :
            sRGBToLinear(selected.rgb);
        color = mix(color, overlay, selected.a);
    }
    OutColor = vec4(linearTosRGB(color), InColor.a);
}
