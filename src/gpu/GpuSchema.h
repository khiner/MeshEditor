#pragma once

#include "state/Slot.h"

struct Transform;
struct ViewportTheme;
struct WorkspaceLights;

using GpuSchema = state::Schema<state::GpuBase, state::GpuCount, Transform, ViewportTheme, WorkspaceLights>;
template<typename T>
    requires GpuSchema::Has<T>
inline constexpr state::TypeId state::Slot<T> = GpuSchema::Of<T>();
