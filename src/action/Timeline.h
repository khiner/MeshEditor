#pragma once

#include "action/Variant.h"

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

using Actions = entt::type_list<TogglePlay, SetFrame, SetStartFrame, SetEndFrame, JumpToStart, JumpToEnd>;
using Action = detail::VariantFromT<Actions>;
} // namespace action::timeline
