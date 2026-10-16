#pragma once

#include <vector>

namespace state {
struct Scene;
} // namespace state
namespace action {
struct Errors {
    std::vector<std::string> Messages;
    bool OnlyUnsupportedExtensions{false};
};
void Fail(state::Scene &, std::string, bool unsupported_extensions = false);
} // namespace action
