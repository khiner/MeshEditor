#pragma once

#include "gpu/Types.h"

struct TextureInfo {
    uint32_t Slot DEFAULT(InvalidSlot);
    uint32_t TexCoord DEFAULT();
    vec2 UvOffset DEFAULT(0);
    vec2 UvScale DEFAULT(1);
    float UvRotation DEFAULT(0);
};
#ifndef __METAL_VERSION__
#include "Field.h"
template<> inline constexpr field::SpecTable<TextureInfo> Specs<TextureInfo>{
    {&TextureInfo::Slot, {.Label = "Texture", .Help = "Texture for this property. None uses the factor alone."}},
    {&TextureInfo::TexCoord, {.Min = 0, .Max = 3, .Label = "UV set", .Help = "Texture coordinate set"}},
    {&TextureInfo::UvOffset, {.Speed = 0.01f, .Label = "UV offset", .Help = "Translation of the texture coordinates, applied after rotation and scale"}},
    {&TextureInfo::UvScale, {.Speed = 0.01f, .Label = "UV scale", .Help = "Scale of the texture coordinates"}},
    {&TextureInfo::UvRotation, {.Speed = 0.01f, .Unit = FieldUnit::Radians, .Label = "UV rotation", .Help = "Counterclockwise rotation of the texture coordinates"}},
};
#endif
