#pragma once

#include "gpu/TextureInfo.h"

struct Volume {
    float ThicknessFactor DEFAULT(0);
    vec3 AttenuationColor DEFAULT(1);
    float AttenuationDistance DEFAULT(0);
    TextureInfo ThicknessTexture DEFAULT();
};
#ifndef __METAL_VERSION__
template<> inline constexpr field::SpecTable<Volume> Specs<Volume>{
    {&Volume::ThicknessFactor, {.Min = 0, .Max = 10, .Label = "Thickness", .Help = "Thickness below the surface, multiplied by the texture's green channel"}},
    {&Volume::AttenuationColor, {.Widget = FieldWidget::Color, .Help = "Color of white light after the attenuation distance through the medium"}},
    {&Volume::AttenuationDistance, {.Min = 0, .Speed = 0.01f, .Unit = FieldUnit::Meters, .Widget = FieldWidget::Drag, .Help = "Distance light travels through the medium before reaching the attenuation color. Zero is infinite."}},
    {&Volume::ThicknessTexture, {.Help = "Volume thickness in the green channel"}},
};
#endif
