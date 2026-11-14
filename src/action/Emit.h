#pragma once

#include <utility>

namespace action {
enum class Phase {
    Record, // Apply and record after committing an open gesture.
    Stage, // Apply and retain until commit.
    Cancel, // Cancel the open gesture, then apply without recording.
};

// Where Append puts an action.
enum class Destination {
    Gesture, // The frame's gesture, which Begin opened.
    Step, // The queue of actions the project applies and commits one at a time.
};
// Each domain's action header declares `void Append(Destination, Action)` for its domain variant.
// A leaf action converts to its domain variant, so Append takes leaves too.

// Opens the frame's gesture and returns whether it did.
// Project::Frame accepts one gesture per frame, so a frame's later gestures are dropped.
bool Begin(Phase);
// Adds an action to the gesture Begin opened.
template<typename A> void AddToGesture(A action) { Append(Destination::Gesture, std::move(action)); }
// Emits one action as the frame's gesture.
template<typename A> void Emit(A action, Phase phase = Phase::Record) {
    if (Begin(phase)) AddToGesture(std::move(action));
}
// Emits the actions, in order, as the frame's gesture.
template<typename... As> void Emit(Phase phase, As... actions) {
    if (Begin(phase)) (AddToGesture(std::move(actions)), ...);
}
// Queues the action as its own history step.
// The project applies queued steps at the end of a frame with no open gesture, or when Project::ApplySteps runs.
template<typename A> void Post(A action) { Append(Destination::Step, std::move(action)); }
// Commits an open gesture without emitting another action.
void Commit();
// Discards an open gesture, restoring the state it started from.
void Cancel();
} // namespace action
