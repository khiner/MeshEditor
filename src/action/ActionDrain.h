#pragma once

#include "action/Action.h"

namespace action {
enum class Phase {
    Record, // Apply and record after committing an open gesture.
    Stage, // Apply and retain until commit.
    Cancel, // Apply a revert and discard the pending step.
};

// Contains the first user action, all system actions, and the standalone commit request for a frame.
struct Drained {
    std::optional<std::pair<Action, Phase>> Emitted;
    std::vector<Action> System;
    bool CommitRequested;
};

// Returns and resets the frame's buffered actions and commit request.
Drained Drain();
} // namespace action
