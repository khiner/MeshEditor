#pragma once

#include "action/Action.h"

namespace action {
// Contains at most one ordered user gesture and the standalone commit and cancel requests for a frame.
struct Drained {
    std::optional<std::pair<std::vector<Action>, Phase>> Emitted;
    bool CommitRequested, CancelRequested;
};

// Returns and resets the frame's buffered gesture and requests.
Drained Drain();
// Returns and empties the queue of posted steps.
std::vector<Action> DrainSteps();
} // namespace action
