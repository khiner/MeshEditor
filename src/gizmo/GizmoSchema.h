#pragma once

#include "state/Slot.h"

struct GizmoInteraction;

using GizmoSchema = state::Schema<state::GizmoBase, state::GizmoCount, GizmoInteraction>;
template<typename T>
    requires GizmoSchema::Has<T>
inline constexpr state::TypeId state::Slot<T> = GizmoSchema::Of<T>();
