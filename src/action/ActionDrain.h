#pragma once

#include "action/Action.h"

namespace action {
enum class Phase {
    Record, // Apply, commit any open gesture, and record this as a committed action.
    Stage, // Apply and hold as the gesture's latest step, recording only on commit.
    Cancel, // Apply a revert and discard the held step, recording nothing.
};

// This frame's drained emission: the winning action + its phase (first emit wins, empty if none), and the standalone commit-request flag.
struct Drained {
    std::optional<std::pair<Action, Phase>> Emitted;
    bool CommitRequested;
};

// Return and reset this frame's emitted action and commit-request flag.
Drained Drain();
} // namespace action
