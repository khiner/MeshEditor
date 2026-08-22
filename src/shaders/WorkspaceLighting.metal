#ifndef WORKSPACELIGHTING_MSL
#define WORKSPACELIGHTING_MSL

// Blender-matching solid viewport lighting. Ported from workbench_world_light.bsl.hh.
#include "Bindless.metal"
#include "Varyings.metal"
#include "tonemapping.metal"
#include "VisibilityDecode.metal"

// Approximate Fresnel effect with roughness attenuation.
inline float3 brdf_approx(float3 spec_color, float roughness, float NV) {
    const float fresnel = exp2(-8.35f * NV) * (1.0f - roughness);
    return mix(spec_color, float3(1.0f), fresnel);
}

// Normalized Blinn-Phong specular.
inline float4 blinn_specular(float4 shininess, float4 spec_angle, float4 NL) {
    const float4 normalization_factor = shininess * 0.125f + 1.0f; // (shininess + 8) / (8 * pi)
    return pow(spec_angle, shininess) * NL * normalization_factor;
}

// Wrapped lighting: NL unclamped, w in [0, 1].
inline float4 wrapped_lighting(float4 NL, float4 w) {
    const float4 w_1 = w + 1.0f;
    return clamp((NL + w) / (w_1 * w_1), 0.0f, 1.0f);
}

inline float3 get_world_lighting(const thread Scene &scene, float3 base_color, float roughness, float metallic, float3 N, float3 I) {
    constant WorkspaceLights &lights = scene.Workspace;
    float3 specular_color, diffuse_color;
    if (lights.UseSpecular != 0u) {
        diffuse_color = mix(base_color, float3(0.0f), metallic);
        specular_color = mix(float3(0.05f), base_color, metallic);
    } else {
        diffuse_color = base_color;
        specular_color = float3(0.0f);
    }

    float3 specular_light = float3(lights.AmbientColor);
    float3 diffuse_light = float3(lights.AmbientColor);
    const float4 wrap = float4(
        lights.Lights[0].Wrap, lights.Lights[1].Wrap, lights.Lights[2].Wrap, lights.Lights[3].Wrap
    );

    if (lights.UseSpecular != 0u) {
        const float3 R = -reflect(I, N);
        float4 spec_angle, spec_NL, wrapped_NL;
        for (int i = 0; i < 4; i++) {
            const float3 L = float3(lights.Lights[i].Direction);
            const float3 half_dir = normalize(L + I);
            wrapped_NL[i] = dot(L, R);
            spec_angle[i] = clamp(dot(half_dir, N), 0.0f, 1.0f);
            spec_NL[i] = clamp(dot(L, N), 0.0f, 1.0f);
        }

        float4 gloss = float4(1.0f - roughness);
        // Reduce gloss for smooth light, which stands in for a bigger light.
        gloss *= 1.0f - wrap;
        const float4 shininess = exp2(10.0f * gloss + 1.0f);
        float4 spec_light = blinn_specular(shininess, spec_angle, spec_NL);

        // Stand in for environment light.
        const float4 spec_env = wrapped_lighting(wrapped_NL, mix(wrap, float4(1.0f), roughness));
        spec_light = mix(spec_light, spec_env, wrap * wrap);

        for (int i = 0; i < 4; i++) {
            specular_light += spec_light[i] * float3(lights.Lights[i].SpecularColor);
        }

        specular_color = brdf_approx(specular_color, roughness, clamp(dot(N, I), 0.0f, 1.0f));
    }
    specular_light *= specular_color;

    // Diffuse: four lights under wrapped lighting.
    float4 diff_NL;
    for (int i = 0; i < 4; i++) diff_NL[i] = dot(float3(lights.Lights[i].Direction), N);

    const float4 diff_light = wrapped_lighting(diff_NL, wrap);
    for (int i = 0; i < 4; i++) diffuse_light += diff_light[i] * float3(lights.Lights[i].DiffuseColor);

    // Energy conservation: reduce diffuse by specular energy.
    const float spec_energy = dot(specular_color, float3(0.33333f));
    diffuse_light *= diffuse_color * (1.0f - spec_energy);

    return diffuse_light + specular_light;
}

inline float4 ShadeWorkspace(MeshVaryings in, const thread Scene &scene, constant SceneViewUBO &view) {
    const float3x3 view_rotation = view.ViewRotation.Unpack();
    // View-space normal and view direction, matching Blender's camera-relative lighting default.
    float3 N = normalize(view_rotation * ShadingWorldNormal(in));
    const float3 view_dir = float3(view.CameraPosition) - in.WorldPosition;
    N = faceforward(N, -(view_rotation * view_dir), N);
    const float3 I = normalize(view_rotation * view_dir);

    // Blender's workbench defaults: metallic 0, roughness 0.4.
    // pack_data() applies the Disney roughness remap (sqrt) before the shader sees it.
    float3 color = get_world_lighting(scene, in.Color.rgb, sqrt(0.4f), 0.0f, N, I);
    const uint overlay_flags = in.FaceOverlayFlags & 3u;
    if (overlay_flags != 0u) {
        // Theme colors are sRGB, so they convert to linear for blending.
        constant ViewportThemeColors &colors = scene.Theme.Colors;
        const bool is_edit_face = view.InteractionMode == InteractionMode_Edit && view.EditElement == Element_Face;
        const float4 selected = is_edit_face ? float4(colors.FaceSelected) : float4(colors.FaceSelectedIncidental);
        const float3 overlay = (overlay_flags & 2u) != 0u ?
            mix(sRGBToLinear(selected.rgb), sRGBToLinear(float4(colors.ElementActive).rgb), 0.5f) :
            sRGBToLinear(selected.rgb);
        color = mix(color, overlay, selected.a);
    }
    return float4(color, in.Color.a);
}

fragment float4 WorkspaceVisibilityFragment(
    QuadVaryings quad [[stage_in]],
    texture2d<uint, access::read> visibility [[texture(0)]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant VisibilityShadingPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const DecodedVisibility decoded = DecodeWorkspaceVisibilityId(
        visibility.read(uint2(quad.Position.xy)).r, quad.Position.xy,
        bindless, view, theme, workspace, pc
    );
    if (!decoded.Valid) discard_fragment();
    const Scene scene{bindless, view, theme, workspace};
    return ShadeWorkspace(decoded.V, scene, view);
}

#endif
