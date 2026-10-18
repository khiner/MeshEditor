#pragma once

#include "state/Slot.h"

struct SurfaceChanges;
struct SurfaceRelief;

using AudioSurfaceSchema = state::Schema<state::AudioSurfaceBase, state::AudioSurfaceCount, SurfaceChanges, SurfaceRelief>;
template<typename T>
    requires AudioSurfaceSchema::Has<T>
inline constexpr state::TypeId state::Slot<T> = AudioSurfaceSchema::Of<T>();
