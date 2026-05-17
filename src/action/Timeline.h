#pragma once

#include <variant>

namespace action::timeline {
struct TogglePlay {};
struct SetFrame {
    int Frame;
};
struct SetStartFrame {
    int Frame;
};
struct SetEndFrame {
    int Frame;
};
struct JumpToStart {};
struct JumpToEnd {};

using Actions = std::variant<TogglePlay, SetFrame, SetStartFrame, SetEndFrame, JumpToStart, JumpToEnd>;
using Action = Actions;
} // namespace action::timeline
