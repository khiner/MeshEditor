#pragma once

#include "state/Slot.h"

struct Transform;
struct ViewportTheme;

using GpuSchema = state::Schema<state::GpuBase, state::GpuCount, Transform, ViewportTheme>;
template<typename T>
    requires GpuSchema::Has<T>
inline constexpr state::TypeId state::Slot<T> = GpuSchema::Of<T>();
