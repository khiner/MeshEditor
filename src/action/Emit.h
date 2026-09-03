#pragma once

#include <cstddef>

namespace action {
// The first user action emitted per frame is buffered for ApplyEmitted.
// A gesture consists of EmitStaged calls followed by commit or cancel.

// Applies and records at frame end after committing an open gesture.
template<typename ActionType> void Emit(ActionType);
// Applies and records a system-generated action at frame end in addition to a user action.
template<typename ActionType> void EmitSystem(ActionType);
// Applies a preview step and records only the latest step when the gesture commits.
template<typename ActionType> void EmitStaged(ActionType);
// Applies a revert and discards the uncommitted gesture step.
template<typename ActionType> void EmitCancel(ActionType);
// Commits an open gesture without emitting another action.
void Commit();

size_t ActionSize();
} // namespace action
