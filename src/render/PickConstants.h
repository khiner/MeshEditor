#pragma once

#include <cstdint>

// Pixel radii for object/element picking. Also size the GPU element-pick buffers.
constexpr uint32_t
    ObjectSelectRadiusPx = 15,
    ElementSelectRadiusPx = 50,
    ElementPickDiameterPx = ElementSelectRadiusPx * 2 + 1,
    ElementPickPixelCount = ElementPickDiameterPx * ElementPickDiameterPx;
