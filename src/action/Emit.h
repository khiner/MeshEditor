#pragma once

#include <entt/entity/fwd.hpp>

#include <cstddef>

namespace action {
// All of these buffer at most one action per frame (first emitted wins), applied in ApplyEmitted.
// A gesture (gizmo drag, box-select, slider) is `EmitStaged`* steps bracketed by a commit or cancel.

// Apply at end of frame and record. Commits any open gesture first, so gesture terminals are plain Emit.
template<typename ActionType> void Emit(ActionType);
// A live-preview gesture step: apply for feedback, hold as the gesture's latest step (superseding the
// prior), and record only when the gesture commits.
template<typename ActionType> void EmitStaged(ActionType);
// Abort the open gesture: apply a revert live, discard the held step, record nothing.
template<typename ActionType> void EmitCancel(ActionType);
// Commit the open gesture (flush its held step) without emitting an action — for terminal-less gestures.
void Commit();

// Apply this frame's buffered action / gesture transition (if any).
void ApplyEmitted(entt::registry &, entt::entity viewport);

std::size_t ActionSize();
} // namespace action
