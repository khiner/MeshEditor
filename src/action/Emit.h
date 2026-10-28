#pragma once

namespace action {
enum class Phase {
    Record, // Apply and record after committing an open gesture.
    Stage, // Apply and retain until commit.
    Cancel, // Cancel the open gesture, then apply without recording.
};

// A single action is a one-action gesture; Project::Frame accepts at most one emitted gesture per frame.
template<typename ActionType> void Emit(ActionType, Phase = Phase::Record);
// Commits an open gesture without emitting another action.
void Commit();
// Discards an open gesture, restoring the state it started from.
void Cancel();
} // namespace action
