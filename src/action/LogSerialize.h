#pragma once
#include "action/Action.h"
#include <iosfwd>
#include <limits>
#include <vector>

namespace action {
// Each record contains a uint32 byte length followed by its domain index, leaf index, and payload.
void SerializeAction(const Action &, std::ostream &);
namespace detail {
bool ReadAction(std::istream &, Action &, std::vector<std::byte> &);
}

// Streams actions with bounded memory and stops at a truncated or corrupt record.
void StreamActions(std::istream &in, auto &&on_action, uint64_t count = std::numeric_limits<uint64_t>::max()) {
    std::vector<std::byte> bytes;
    for (uint64_t i = 0; i < count; ++i) {
        Action a;
        if (!detail::ReadAction(in, a, bytes)) return;
        on_action(std::move(a));
    }
}
} // namespace action
