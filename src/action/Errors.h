#pragma once

#include <string>
#include <vector>

namespace action {
// Stores action-handler failures for the application to drain each frame.
struct Errors {
    std::vector<std::string> Messages;
};
} // namespace action
