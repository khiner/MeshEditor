#pragma once

#include <string>
#include <vector>

namespace action {
// Side channel for action handlers to report failures, since an action's Apply returns nothing.
// Stored in the registry context, appended to by handlers, and drained/surfaced by the app each frame.
struct Errors {
    std::vector<std::string> Messages;
};
} // namespace action
