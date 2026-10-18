#pragma once

#include "state/Slot.h"

struct PendingObjectRemovals;
struct PendingSlotRemovals;

using ObjectSchema = state::Schema<state::ObjectBase, state::ObjectCount, PendingObjectRemovals, PendingSlotRemovals>;
template<typename T>
    requires ObjectSchema::Has<T>
inline constexpr state::TypeId state::Slot<T> = ObjectSchema::Of<T>();
