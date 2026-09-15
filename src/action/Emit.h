#pragma once

namespace action {
enum class Phase {
    Record, // Apply and record after committing an open gesture.
    Stage, // Apply and retain until commit.
    Cancel, // Cancel the open gesture, then apply without recording.
};

// Project::Frame applies the first user action emitted in each frame.
template<typename ActionType> void Emit(ActionType, Phase = Phase::Record);
// Applies and records a system-generated action at frame end in addition to a user action.
template<typename ActionType> void EmitSystem(ActionType);
// Commits an open gesture without emitting another action.
void Commit();
// Discards an open gesture, restoring the state it started from.
void Cancel();
} // namespace action
