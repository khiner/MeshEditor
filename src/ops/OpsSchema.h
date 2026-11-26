#pragma once

#include "state/Slot.h"

namespace ops {
struct Drivers;

using OpsSchema = state::Schema<state::OpsBase, state::OpsCount, Drivers>;
} // namespace ops
template<typename T>
    requires ops::OpsSchema::Has<T>
inline constexpr state::TypeId state::Slot<T> = ops::OpsSchema::Of<T>();
