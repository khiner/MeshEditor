#pragma once

#include "gpu/Types.h"

struct TextureInfo {
    uint32_t Slot DEFAULT(InvalidSlot);
    uint32_t TexCoord DEFAULT();
    vec2 UvOffset DEFAULT(0);
    vec2 UvScale DEFAULT(1);
    float UvRotation DEFAULT(0);
};
static_assert(sizeof(TextureInfo) == 28, "TextureInfo size");
#ifndef __METAL_VERSION__
#include "Field.h"
template<> inline constexpr FieldSpec Spec<TextureInfo, "UvOffset">{.Speed = 0.01f};
template<> inline constexpr FieldSpec Spec<TextureInfo, "UvScale">{.Speed = 0.01f};
template<> inline constexpr FieldSpec Spec<TextureInfo, "UvRotation">{.Speed = 0.01f};
#endif
