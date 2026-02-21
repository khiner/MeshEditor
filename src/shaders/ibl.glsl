// Adapted from KhronosGroup/glTF-Sample-Renderer (ibl.glsl), pulled 2026-02-21.

#ifndef IBL_GLSL
#define IBL_GLSL

mat3 getEnvRotation() {
    const float s = sin(SceneViewUBO.EnvRotationRadians);
    const float c = cos(SceneViewUBO.EnvRotationRadians);
    return mat3(
        c, 0.0, -s,
        0.0, 1.0, 0.0,
        s, 0.0, c
    );
}

vec3 getDiffuseLight(vec3 n) {
    vec4 texture_sample = texture(
        CubeSamplers[nonuniformEXT(SceneViewUBO.DiffuseEnvSamplerSlot)],
        getEnvRotation() * n
    );
    texture_sample.rgb *= SceneViewUBO.EnvIntensity;
    return texture_sample.rgb;
}

vec4 getSpecularSample(vec3 reflection, float lod) {
    vec4 texture_sample = textureLod(
        CubeSamplers[nonuniformEXT(SceneViewUBO.SpecularEnvSamplerSlot)],
        getEnvRotation() * reflection,
        lod
    );
    texture_sample.rgb *= SceneViewUBO.EnvIntensity;
    return texture_sample;
}

vec3 getIBLGGXFresnel(vec3 n, vec3 v, float roughness, vec3 F0, float specularWeight) {
    const float NdotV = clamp(dot(n, v), 0.0, 1.0);
    const vec2 brdf_sample_point = clamp(vec2(NdotV, roughness), vec2(0.0), vec2(1.0));
    const vec2 f_ab = texture(Samplers[nonuniformEXT(SceneViewUBO.BrdfLutSamplerSlot)], brdf_sample_point).rg;

    const vec3 Fr = max(vec3(1.0 - roughness), F0) - F0;
    const vec3 k_S = F0 + Fr * pow(1.0 - NdotV, 5.0);
    const vec3 FssEss = specularWeight * (k_S * f_ab.x + f_ab.y);

    const float Ems = (1.0 - (f_ab.x + f_ab.y));
    const vec3 F_avg = specularWeight * (F0 + (1.0 - F0) / 21.0);
    const vec3 FmsEms = Ems * FssEss * F_avg / (1.0 - F_avg * Ems);

    return FssEss + FmsEms;
}

vec3 getIBLRadianceGGX(vec3 n, vec3 v, float roughness) {
    const uint mip_count = max(SceneViewUBO.SpecularEnvMipCount, 1u);
    const float lod = roughness * float(mip_count - 1u);
    const vec3 reflection = normalize(reflect(-v, n));
    return getSpecularSample(reflection, lod).rgb;
}

#endif
