#pragma once

#include <entt/entity/fwd.hpp>

namespace action::timeline {
// Enter presentation view (material-preview shading, overlays off) and start playback.
struct StartPresentation {};

// Frame pins CurrentFrame on apply, so a recorded stop replays to the same frame.
struct TogglePlay {
    int Frame;
};
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
    float PixelsPerFrame, ViewCenterFrame;
};

using Action = std::variant<TogglePlay, StartPresentation, SetFrame, SetStartFrame, SetEndFrame, JumpToStart, JumpToEnd, SetView>;

void Apply(entt::registry &, entt::entity viewport, const Action &);
} // namespace action::timeline
