#pragma once

#include "gpu/TextureInfo.h"

struct Anisotropy {
    float Strength DEFAULT(0);
    float Rotation DEFAULT(0);
    TextureInfo Texture DEFAULT();
};
#ifndef __METAL_VERSION__
#include <numbers>

template<> inline constexpr field::SpecTable<Anisotropy> Specs<Anisotropy>{
    {&Anisotropy::Strength, {.Min = 0, .Max = 1, .Help = "Anisotropy strength, multiplied by the texture's blue channel"}},
    {&Anisotropy::Rotation, {.Min = 0, .Max = 2 * std::numbers::pi_v<float>, .Unit = FieldUnit::Radians, .Help = "Counterclockwise rotation of the anisotropy direction from the tangent"}},
    {&Anisotropy::Texture, {.Help = "Direction in the red and green channels, and strength in the blue channel"}},
};
#endif
