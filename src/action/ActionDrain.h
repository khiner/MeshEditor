#pragma once

#include "action/Action.h"

namespace action {
// Contains at most one ordered user gesture and the standalone commit and cancel requests for a frame.
struct Drained {
    std::optional<std::pair<std::vector<Action>, Phase>> Emitted;
    bool CommitRequested, CancelRequested;
};

// Returns and resets the frame's buffered actions and requests.
Drained Drain();
} // namespace action
