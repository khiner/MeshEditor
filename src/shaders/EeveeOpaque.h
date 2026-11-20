/* SPDX-FileCopyrightText: 2017-2026 Blender Authors
 * SPDX-License-Identifier: GPL-2.0-or-later
 * Blender 222853b0d7e: eevee_nodetree_lib::F_brdf_multi_scatter,
 * eevee_bxdf_diffuse and eevee_ray_generate. glTF closure adapter.
 */
#pragma once
#include "EeveeGlass.h"
namespace eevee_opaque {
struct Closure {
    float3 N, Reflection, Diffuse;
    float Roughness;
};
inline float3 fresnel(float3 f0, float3 f90, float2 lut) {
    const float3 FssEss = f0 * lut.x + f90 * lut.y;
    const float Ems = 1.0f - (lut.x + lut.y);
    const float3 Favg = f0 + (f90 - f0) / 21.0f;
    return FssEss / (1.0f - Ems * Favg);
}
inline Closure prepare(texture2d_array<float> table, float3 n, float3 v, float roughness, float3 base, float metallic, float3 f0, float specular, float ao) {
    const float2 lut = eevee_glass::lut2(table, float2(roughness, sqrt(saturate(1.0f - dot(n, v)))), 1u).xy;
    const float3 dielectric = fresnel(f0 * specular, float3(specular), lut);
    return {n, mix(dielectric, fresnel(base, float3(1), lut), metallic) * ao, base * (1.0f - metallic) * (1.0f - dielectric) * ao, roughness};
}
inline float pdf(float3 n, float3 v, float roughness, uint lobe, float3 direction) {
    if (lobe) return max(dot(n, direction), 0.0f) / M_PI_F;
    return eevee_microfacet::bxdf_ggx_eval_reflection(n, direction, v, roughness * roughness, true).pdf;
}
inline float4 generate(float3 n, float3 v, float roughness, uint lobe, float2 noise) {
    const float3x3 frame = RayFrame(n);
    const float phi = 2.0f * M_PI_F * noise.y;
    if (lobe) {
        const float radius = sqrt(noise.x), z = sqrt(1.0f - noise.x);
        return float4(frame * float3(radius * cos(phi), radius * sin(phi), z), M_PI_F / max(z, 1e-8f));
    }
    const auto sample = eevee_microfacet::bxdf_ggx_sample_reflection(float3(1.0f - noise.x * 0.92f, cos(phi), sin(phi)), transpose(frame) * v, roughness * roughness, true);
    return float4(frame * sample.direction, sample.pdf > 0.0f ? 1.0f / sample.pdf : 0.0f);
}
} // namespace eevee_opaque
