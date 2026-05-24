#pragma once

#include <entt/entity/fwd.hpp>

namespace action::timeline {
struct TogglePlay {};
struct Play {};
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
struct SetView {
    float PixelsPerFrame;
    float ViewCenterFrame;
};

using Actions = std::variant<TogglePlay, Play, SetFrame, SetStartFrame, SetEndFrame, JumpToStart, JumpToEnd, SetView>;
using Action = Actions;

void Apply(entt::registry &, entt::entity viewport, const Action &);
} // namespace action::timeline
