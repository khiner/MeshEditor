#pragma once

#include "numeric/vec4.h"

#include <cstdint>

constexpr numeric::vec4 Rgba8(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255) { return {r / 255.f, g / 255.f, b / 255.f, a / 255.f}; }
