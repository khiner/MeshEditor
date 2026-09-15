#pragma once

#include "action/Action.h"
#include "action/Emit.h"

namespace action {
// Contains the first user action, all system actions, and the standalone commit and cancel requests for a frame.
struct Drained {
    std::optional<std::pair<Action, Phase>> Emitted;
    std::vector<Action> System;
    bool CommitRequested, CancelRequested;
};

// Returns and resets the frame's buffered actions and requests.
Drained Drain();
} // namespace action
