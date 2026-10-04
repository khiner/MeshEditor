#pragma once

#include <string>
#include <vector>

namespace state {
struct Scene;
}
namespace action {
// Stores action-handler failures for the application to drain each frame.
struct Errors {
    std::vector<std::string> Messages;
    bool OnlyUnsupportedExtensions{false};
};
void Fail(state::Scene &, std::string, bool unsupported_extensions = false);
} // namespace action
