#pragma once

#include "gpu/Types.h"
#ifndef __METAL_VERSION__
#include "Field.h"
#endif

struct Transform {
    vec3 P DEFAULT();
    quat R DEFAULT(1, 0, 0, 0);
    vec3 S DEFAULT(1, 1, 1);
#ifndef __METAL_VERSION__
    bool operator==(const Transform &) const = default;
#endif
};
#ifndef __METAL_VERSION__
template<> inline constexpr field::SpecTable<Transform> Specs<Transform>{
    {&Transform::P, {.Speed = 0.01f, .Unit = FieldUnit::Meters, .Label = "Position", .Help = "Translation relative to the parent"}},
    {&Transform::R, {.Label = "Rotation", .Help = "Orientation relative to the parent"}},
    {&Transform::S, {.Min = 0.01, .Max = 10, .Speed = 0.01f, .Widget = FieldWidget::Drag, .Label = "Scale", .Help = "Scale along each local axis"}},
};
#endif
