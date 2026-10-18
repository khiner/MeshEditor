#pragma once

#include "state/Slot.h"

namespace project {
struct Assets;
struct Project;

using ProjectSchema = state::Schema<state::ProjectBase, state::ProjectCount, Assets, Project *>;
} // namespace project
template<typename T>
    requires project::ProjectSchema::Has<T>
inline constexpr state::TypeId state::Slot<T> = project::ProjectSchema::Of<T>();
