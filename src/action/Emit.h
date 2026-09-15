#pragma once

namespace action {
// Project::Frame applies the first user action emitted in each frame.

// Applies and records at frame end after committing an open gesture.
template<typename ActionType> void Emit(ActionType);
// Applies and records a system-generated action at frame end in addition to a user action.
template<typename ActionType> void EmitSystem(ActionType);
// Queue an update to the current gesture.
template<typename ActionType> void EmitStaged(ActionType);
// Queue cancellation of the current gesture.
template<typename ActionType> void EmitCancel(ActionType);
// Commits an open gesture without emitting another action.
void Commit();
} // namespace action
