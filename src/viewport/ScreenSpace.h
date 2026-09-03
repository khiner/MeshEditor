#pragma once

#include "numeric/vec2.h"

// Clip-space Y points up.
// Pixel and texture Y point down.
constexpr vec2 NdcToUv(vec2 ndc) { return {ndc.x * 0.5f + 0.5f, 0.5f - ndc.y * 0.5f}; }
constexpr vec2 UvToNdc(vec2 uv) { return {uv.x * 2.f - 1.f, 1.f - uv.y * 2.f}; }
