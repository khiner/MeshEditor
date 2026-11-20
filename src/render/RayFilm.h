/* SPDX-FileCopyrightText: 2023 Blender Authors
 * SPDX-License-Identifier: GPL-2.0-or-later
 * EEVEE sampling and film filter adapters.
 */
#pragma once
#include "gpu/RayFilmPushConstants.h"
#include <cmath>

// EEVEE sampling.cc Halton(2,3); phase shifted so the first render is pixel-centered.
inline vec2 RayFilmJitter(uint32_t index) {
    const auto halton = [](uint32_t n, uint32_t base) {
        double value = 0, weight = 1;
        for (; n; n /= base) {
            weight /= base;
            value += double(n % base) * weight;
        }
        return value;
    };
    return {float(halton(index + 1u, 2u) - .5), float(std::fmod(halton(index + 1u, 3u) + 1. / 6., 1.) - .5)};
}
// Blender 222853b0d7e film_filter_weight and Film::update_sample_table, radius 1.5 pixels.
inline RayFilmPushConstants RayFilmFilter(vec2 jitter, float previous_weight, bool filter = true) {
    RayFilmPushConstants result{.PreviousWeight = previous_weight};
    for (int y = -1; y <= 1; ++y)
        for (int x = -1; x <= 1; ++x) {
            const float dx = float(x) - jitter.x, dy = float(y) - jitter.y, radius2 = 2.25f;
            const float distance2 = dx * dx + dy * dy;
            const float weight = filter ? (distance2 < radius2 ? std::exp(-0.5f / (.284f * .284f) * distance2 / radius2) : 0.f) : float(x == 0 && y == 0);
            result.Weights[(y + 1) * 3 + x + 1] = weight;
            result.Weight += weight;
        }
    return result;
}
