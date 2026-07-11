#pragma once

#include <cstddef>

namespace action {
// A UI pass emits at most one user action per frame (first emitted wins), buffered and applied in `ApplyEmitted`.
// A gesture (gizmo drag, box-select, slider) is `EmitStaged`* steps bracketed by a commit or cancel.

// Apply at end of frame and record. Commits any open gesture first, so gesture terminals are plain Emit.
template<typename ActionType> void Emit(ActionType);
// System-generated action (e.g. a background job's result): applied and recorded at end of frame
// in addition to any user action. Never dropped.
template<typename ActionType> void EmitSystem(ActionType);
// A live-preview gesture step: apply for feedback, hold as the gesture's latest step (superseding the
// prior), and record only when the gesture commits.
template<typename ActionType> void EmitStaged(ActionType);
// Abort the open gesture: apply a revert live, discard the held step, record nothing.
template<typename ActionType> void EmitCancel(ActionType);
// Commit the open gesture (flush its held step) without emitting an action — for terminal-less gestures.
void Commit();

size_t ActionSize();
} // namespace action
