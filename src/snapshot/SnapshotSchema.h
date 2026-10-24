#pragma once

#include "state/Slot.h"

namespace snapshot {
struct Retention;

using SnapshotSchema = state::Schema<state::SnapshotBase, state::SnapshotCount, Retention>;
} // namespace snapshot
template<typename T>
    requires snapshot::SnapshotSchema::Has<T>
inline constexpr state::TypeId state::Slot<T> = snapshot::SnapshotSchema::Of<T>();
