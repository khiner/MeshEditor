/* SPDX-FileCopyrightText: 2017-2026 Blender Authors
 * SPDX-License-Identifier: GPL-2.0-or-later
 * Metal port of Blender 222853b0d7e: eevee_bxdf_microfacet, eevee_bxdf_lut_lib,
 * eevee_nodetree_lib::bsdf_lut, eevee_ray_generate and eevee_thickness_lib.
 */
#pragma once
#include "EeveeMicrofacet.h"
#include "RayTracing.h"
namespace eevee_glass {
inline float square(float x) { return x * x; }
inline float4 lut2(texture2d_array<float> table, float2 uv, uint layer) {
    constexpr sampler linear_sampler(coord::normalized, address::clamp_to_edge, filter::linear);
    return table.sample(linear_sampler, uv * (63.0f / 64.0f) + (0.5f / 64.0f), layer, level(0));
}
inline float4 lut3(texture2d_array<float> table, float2 uv, float roughness) {
    const float z = saturate(roughness) * 16.0f;
    return mix(lut2(table, uv, 2u + min(uint(z), 15u)), lut2(table, uv, 2u + min(uint(z) + 1u, 15u)), fract(z));
}
inline void weights(texture2d_array<float> table, float nv, float roughness, float ior, float3 tint, float3 f0_color, float3 f90, thread float3 &reflection, thread float3 &transmission) {
    if (ior == 1.0f) {
        reflection = 0.0f;
        transmission = tint;
        return;
    }
    const float f0 = (ior - 1.0f) / (ior + 1.0f);
    float3 data;
    if (ior > 1.0f) {
        if (all(f90 == float3(1))) f90 = float3(saturate(2.33f / 0.33f * f0));
        data.xy = lut2(table, float2(roughness, sqrt(saturate(1.0f - nv))), 1u).xy;
        data.z = lut3(table, float2(sqrt(f0), sqrt(saturate(1.0f - nv))), roughness).w;
    } else {
        const float critical = sqrt(1.0f - ior * ior), delta = nv - critical;
        const float y = delta / (delta > 0.0f ? 1.0f - critical : critical);
        data = lut3(table, float2(ior * ior, 0.5f * y + 0.5f), roughness).xyz;
    }
    reflection = f0_color * data.x + f90 * data.y;
    transmission = (1.0f - f0_color) * data.z * tint;
    const float ess = f0 * f0 * data.x + data.y + (1.0f - f0 * f0) * data.z;
    const float3 average = all(tint == float3(0)) ? f0_color + (f90 - f0_color) / 21.0f : tint;
    const float3 compensation = 1.0f / (1.0f - (1.0f - ess) * average);
    reflection *= compensation;
    transmission *= compensation;
}
inline float apparent_roughness(float roughness, float ior) {
    return roughness * sqrt(ior > 1.0f ? 1.0f - 1.0f / ior : saturate(1.0f - ior) * 0.64f);
}
inline float3 dominant_transmission(float3 n, float3 v, float ior, float roughness) {
    const float m = roughness * roughness, smooth = 1.0f - m;
    return normalize(mix(-n, refract(-v, n, 1.0f / ior), smooth * (sqrt(smooth) + m)));
}
struct Closure {
    float3 N, V, Position, Reflection, Transmission;
    float Roughness, Ior, Thickness, Fraction;
    void amend(thread float3 &n, thread float3 &v, thread float3 &origin) const {
        if (Thickness == 0.0f) return;
        const float3 l = dominant_transmission(N, V, Ior, apparent_roughness(Roughness, Ior));
        const auto hit = eevee_microfacet::Thickness{Thickness}.shape_intersect(N, l);
        n = -hit.hit_N;
        origin += hit.hit_P;
        v = -l;
    }
    float transmission_distance() const {
        if (Thickness == 0.0f) return 0.0f;
        const float3 l = dominant_transmission(N, V, Ior, apparent_roughness(Roughness, Ior));
        const float cosine = dot(l, -N);
        return Thickness > 0.0f ? cosine * Thickness : -Thickness / cosine;
    }
    float roughness(uint lobe) const {
        if (!lobe) return Roughness;
        if (Thickness == 0.0f) return saturate(Roughness * sqrt(max(0.0f, 3.4f * (Ior - 1.0f) * square(Ior - 0.5f) / (Ior * Ior * Ior))));
        return apparent_roughness(Roughness, Ior);
    }
};
inline float pdf(Closure cl, uint lobe, float3 direction) {
    float3 n = cl.N, v = cl.V, p = cl.Position;
    if (!lobe) return eevee_microfacet::bxdf_ggx_eval_reflection(n, direction, v, cl.Roughness * cl.Roughness, true).pdf;
    if (cl.Thickness == 0.0f) {
        const float rough = cl.roughness(1u);
        return eevee_microfacet::bxdf_ggx_eval_reflection(n, direction - 2.0f * dot(direction, n) * n, v, rough * rough, true).pdf;
    }
    cl.amend(n, v, p);
    return eevee_microfacet::bxdf_ggx_eval_refraction(n, direction, v, cl.Roughness * cl.Roughness, cl.Ior, {cl.Thickness}, true).pdf;
}
inline float4 generate(Closure cl, uint lobe, float2 noise, thread float3 &origin) {
    float3 n = cl.N, v = cl.V;
    origin = cl.Position;
    const bool refractive = lobe && cl.Thickness != 0.0f;
    if (refractive) cl.amend(n, v, origin);
    const float3x3 frame = RayFrame(n);
    v = transpose(frame) * v;
    const float rough = lobe && !refractive ? cl.roughness(lobe) : cl.Roughness;
    const float alpha = rough * rough;
    const float phi = 2.0f * M_PI_F * noise.y;
    const float3 cylinder(1.0f - noise.x * 0.92f, cos(phi), sin(phi));
    eevee_microfacet::BsdfSample sample;
    if (lobe && cl.Ior == 1.0f) {
        sample = {-v, 1e6f};
    } else if (refractive) {
        sample = eevee_microfacet::bxdf_ggx_sample_refraction(cylinder, v, alpha, cl.Ior, {cl.Thickness}, true);
    } else {
        sample = eevee_microfacet::bxdf_ggx_sample_reflection(cylinder, v, alpha, true);
        if (lobe) sample.direction.z = -sample.direction.z;
    }
    return float4(frame * sample.direction, isfinite(sample.pdf) && sample.pdf > 0.0f ? 1.0f / sample.pdf : 0.0f);
}
} // namespace eevee_glass
