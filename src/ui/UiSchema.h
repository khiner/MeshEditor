#pragma once

#include "state/Slot.h"

struct InterfaceTheme;

using UiSchema = state::Schema<state::UiBase, state::UiCount, InterfaceTheme>;
template<typename T>
    requires UiSchema::Has<T>
inline constexpr state::TypeId state::Slot<T> = UiSchema::Of<T>();
