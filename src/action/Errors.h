#pragma once

#include "action/ActionSchema.h"

#include <vector>

namespace state {
struct Registry;
} // namespace state
namespace action {
struct Errors {
    std::vector<std::string> Messages;
    bool OnlyUnsupportedExtensions{false};
};
void Fail(state::Registry &, std::string, bool unsupported_extensions = false);
} // namespace action
