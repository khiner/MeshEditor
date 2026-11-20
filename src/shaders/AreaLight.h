#pragma once
#include "EeveeLtc.h"
#include "EeveeMicrofacet.h"
#include "punctual.metal"
#include "gpu/Hash.h"

struct AreaEmitter {
    eevee_ltc::LightShape Shape;
    float3 Center, X, Y, Radiance;
    bool Rectangle;
    float2 NoiseRotation;
};

// EEVEE eevee_light.cc prepares world emitter axes and shape radiance once per light.
inline PreparedAreaLight PrepareAreaLight(LightRecord light, Transform world, uint index) {
    PreparedAreaLight result{};
    const float2 half_size = max(float2(light.Size), 0.0f) * abs(float2(world.S.xy)) * 0.5f;
    if (half_size.x * half_size.y < 0.00001f || light.Intensity <= 0.0f) return result;
    const float sign_z = world.S.z < 0.0f ? -1.0f : 1.0f;
    result.Z = quat_rotate(float4(world.R), float3(0, 0, sign_z));
    result.X = quat_rotate(float4(world.R), float3(half_size.x, 0, 0));
    result.Y = quat_rotate(float4(world.R), float3(0, half_size.y * sign_z, 0));
    result.Radiance = eevee_ltc::radiance(light.Intensity, result.X, result.Y, light.Type == LightType::Rectangle);
    result.Valid = 1u;
    result.NoiseRotation = float2(HashMix(index) & 65535u, HashMix(index + 17u) & 65535u) / 65536.0f;
    return result;
}

inline bool PrepareAreaEmitter(const thread Scene &scene, LightRecord light, float3 position, thread AreaEmitter &emitter, uint light_index = InvalidOffset) {
    const uint slot = scene.View.ModelSlotOverride != InvalidSlot ? scene.View.ModelSlotOverride : light.TransformSlotOffset.Slot;
    const Transform world = scene.Models(slot)[light.TransformSlotOffset.Offset];
    emitter.Center = float3(world.P) - position;
    const float falloff = light.Range > 0.0f ? saturate(1.0f - pow(length(emitter.Center) / light.Range, 4.0f)) : 1.0f;
    if (falloff == 0.0f) return false;
    const PreparedAreaLight prepared = light_index != InvalidOffset && scene.View.RayAreaLightSlot != InvalidSlot ?
        BindlessBuffer(PreparedAreaLight, scene.B.Buffer, scene.View.RayAreaLightSlot)[light_index] : PrepareAreaLight(light, world, light_index);
    if (!prepared.Valid || dot(float3(prepared.Z), emitter.Center) <= 0.0f) return false;
    emitter.NoiseRotation = prepared.NoiseRotation;
    emitter.X = prepared.X;
    emitter.Y = prepared.Y;
    emitter.Rectangle = light.Type == LightType::Rectangle;
    emitter.Shape = eevee_ltc::shape(emitter.Center, emitter.X, emitter.Y, emitter.Rectangle);
#ifdef FILTER_AREA_SHADOWS
    // The emitter rotation supplies an orthonormal basis. PrepareAreaLight
    // applies the same mirror sign to Y and Z when scaling the emitter.
    emitter.Shape.N = float3(prepared.Z);
#endif
    emitter.Radiance = float3(light.Color) * (falloff * prepared.Radiance);
    return any(emitter.Radiance > 0.0f);
}

// Preserve a separate compiled boundary around the ellipse cubic solver.
__attribute__((noinline)) float AreaIntegral(texture2d_array<float> table, const thread eevee_ltc::LightShape &shape, float3 L, bool rectangle, eevee_ltc::LTCData data) {
    return rectangle ? eevee_ltc::evaluate_quad(table, shape, L, data) : eevee_ltc::evaluate_disk(table, shape, L, data);
}
inline float AreaDiffuse(texture2d_array<float> table, const thread AreaEmitter &emitter, float3 N, float3 V) {
    return AreaIntegral(table, emitter.Shape, normalize(emitter.Center), emitter.Rectangle, eevee_ltc::diffuse(N, dot(N, V) < 0.0f ? -V : V));
}
inline float AreaGgx(texture2d_array<float> table, const thread AreaEmitter &emitter, float3 N, float3 V, float roughness) {
    return AreaIntegral(table, emitter.Shape, normalize(emitter.Center), emitter.Rectangle, eevee_ltc::ggx(table, N, V, roughness));
}
// EEVEE uses the exit surface and refracted cosine for solid transmission,
// and the original roughness and view cosine for thin-sheet transmission.
inline float AreaTransmission(texture2d_array<float> table, const thread AreaEmitter &emitter, float3 N, float3 V, float roughness, float ior, float thickness) {
    const float perceived = thickness == 0.0f ? roughness : eevee_microfacet::bxdf_ggx_perceived_roughness_transmission(roughness, ior);
    if (thickness != 0.0f) {
        const float3 L = eevee_microfacet::bxdf_ggx_dominant_direction_transmission(N, V, ior, perceived);
        N = -eevee_microfacet::Thickness{thickness}.shape_intersect(N, L).hit_N;
        V = -L;
    }
    const float3 R = thickness == 0.0f ? -V : refract(-V, N, ior);
    if (dot(R, R) == 0.0f) return 0.0f;
    const float nv = dot(-N, R);
    const float2 uv = float2(perceived, sqrt(saturate(1.0f - nv))) * (63.0f / 64.0f) + (0.5f / 64.0f);
    const float4 p = table.sample(eevee_ltc::linear_sampler, uv, 18u, level(0));
    const float3x3 inverse(float3(p.x, 0, p.y), float3(0, 1, 0), float3(p.z, 0, p.w));
    const float3 frame_v = dot(-N, V) < -0.999999f ? -V : V;
    const eevee_ltc::LTCData data{inverse * transpose(eevee_ltc::from_incident_vector(-N, frame_v)), saturate((perceived - 0.15f) * 2.5f), eevee_ltc::LTCFormFactorType::OneSidedCosineSphereClipped};
    return AreaIntegral(table, emitter.Shape, normalize(emitter.Center), emitter.Rectangle, data);
}
