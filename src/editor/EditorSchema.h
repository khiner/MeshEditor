#pragma once

#include "state/Slot.h"

struct AudioChanges;
struct ModalSolveJobs;

using EditorSchema = state::Schema<state::EditorBase, state::EditorCount, AudioChanges, ModalSolveJobs>;
template<typename T>
    requires EditorSchema::Has<T>
inline constexpr state::TypeId state::Slot<T> = EditorSchema::Of<T>();
