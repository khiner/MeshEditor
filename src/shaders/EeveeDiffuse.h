/* SPDX-FileCopyrightText: 2023 Blender Authors
 * SPDX-License-Identifier: GPL-2.0-or-later
 * Blender 222853b0d7e eevee_bxdf_diffuse.bsl.hh: thin/slab Lambertian sampling.
 * The mixture density combines glTF diffuse reflection and transmission weights.
 */
#pragma once
#include <metal_stdlib>
using namespace metal;
namespace eevee_diffuse {
inline float3 sample(float2 noise, bool transmission) {
    const float radius = sqrt(noise.x), phi = 2.0f * M_PI_F * noise.y;
    const float3 direction(radius * cos(phi), radius * sin(phi), sqrt(1.0f - noise.x));
    return transmission ? -direction : direction;
}
inline float pdf(float cosine, float transmission) {
    return ((1.0f - transmission) * max(cosine, 0.0f) + transmission * max(-cosine, 0.0f)) / M_PI_F;
}
} // namespace eevee_diffuse
