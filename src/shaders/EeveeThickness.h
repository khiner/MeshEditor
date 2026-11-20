/* SPDX-FileCopyrightText: 2023 Blender Authors
 * SPDX-License-Identifier: GPL-2.0-or-later
 * Metal port of Blender 222853b0d7e: eevee_thickness_lib.bsl.hh.
 */
#pragma once
#include <metal_stdlib>
namespace eevee_microfacet {
struct ThicknessIsect {
    metal::float3 hit_N, hit_P;
};
// The sign stores the sphere/slab model in closure and reconstruction textures.
struct Thickness {
    float data;
    float value() const { return metal::abs(data); }
    ThicknessIsect shape_intersect(metal::float3 N, metal::float3 L) const {
        const float cosine = metal::dot(L, -N);
        if (data > 0.0f) return {metal::normalize(N + L * (cosine * 2.0f)), L * (cosine * value())};
        return {-N, L * (value() / cosine)};
    }
};
}
